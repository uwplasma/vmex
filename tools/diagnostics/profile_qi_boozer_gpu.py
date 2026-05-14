#!/usr/bin/env python
"""Profile the QI Boozer-objective path without running an optimizer.

This is a developer diagnostic for GPU/CPU comparison.  It runs one VMEC
fixed-boundary solve, then evaluates the differentiable Boozer/QI residual one
or more times on the solved state.  The goal is to isolate Boozer/QI
compile/runtime cost from SciPy trust-region bookkeeping and VMEC replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64, jax, jnp
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasi_isodynamic import quasi_isodynamic_residual_from_state

import vmec_jax as vj


def _block_until_ready(value: Any) -> Any:
    if jax is None:
        return value

    def _block_leaf(leaf):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            return block()
        return leaf

    return jax.tree_util.tree_map(_block_leaf, value)


def _timed(label: str, fn):
    start = time.perf_counter()
    value = fn()
    _block_until_ready(value)
    wall_s = time.perf_counter() - start
    print(f"{label}: {wall_s:.3f} s", flush=True)
    return value, wall_s


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _runtime_info() -> dict[str, Any]:
    if jax is None:
        return {"jax_available": False}
    return {
        "jax_available": True,
        "jax_version": getattr(jax, "__version__", None),
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "examples/data/input.nfp2_QI")
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/qi_boozer_profile.json"))
    parser.add_argument("--solver-device", default=None, help="None/default/cpu/gpu passed to run_fixed_boundary.")
    parser.add_argument("--mpol", type=int, default=6)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--mboz", type=int, default=10)
    parser.add_argument("--nboz", type=int, default=10)
    parser.add_argument("--nphi", type=int, default=61)
    parser.add_argument("--nalpha", type=int, default=13)
    parser.add_argument("--n-bounce", type=int, default=21)
    parser.add_argument("--surfaces", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--repeat", type=int, default=2, help="Number of repeated QI evaluations after VMEC solve.")
    parser.add_argument("--jit-booz", action="store_true", help="Use the jitted Boozer path when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_x64(True)
    input_path = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    solver_device = None if args.solver_device in (None, "", "none", "None") else str(args.solver_device)
    surfaces = np.asarray([float(item) for item in str(args.surfaces).split(",") if item.strip()], dtype=float)

    print("QI Boozer profiler")
    print(f"  input:         {input_path}")
    print(f"  output:        {output_path}")
    print(f"  solver_device: {solver_device}")
    print(f"  runtime:       {_runtime_info()}")

    _cfg, indata = vj.load_config(str(input_path))
    indata = rebuild_indata_with_resolution(indata, mpol=int(args.mpol), ntor=int(args.ntor))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt_input = output_path.parent / f"{input_path.name}.profile_rebuilt"
    vj.write_indata(rebuilt_input, indata)

    run, vmec_wall_s = _timed(
        "vmec fixed-boundary solve",
        lambda: vj.run_fixed_boundary(
            rebuilt_input,
            verbose=False,
            solver_device=solver_device,
        ),
    )
    geom = eval_geom(run.state, run.static)
    signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(run.indata, run.static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(run.static.s))

    qi_times = []
    qi_last = None
    for index in range(max(1, int(args.repeat))):
        qi_last, wall_s = _timed(
            f"QI Boozer residual eval {index + 1}",
            lambda: quasi_isodynamic_residual_from_state(
                state=run.state,
                static=run.static,
                indata=run.indata,
                signgs=signgs,
                flux_local=flux,
                prof_local={"pressure": pressure},
                pressure_local=pressure,
                surfaces=surfaces,
                mboz=int(args.mboz),
                nboz=int(args.nboz),
                nphi=int(args.nphi),
                nalpha=int(args.nalpha),
                n_bounce=int(args.n_bounce),
                softness=2.0e-2,
                branch_width_weight=0.5,
                branch_width_softness=2.0e-2,
                profile_weight=0.1,
                shuffle_profile_weight=1.0,
                shuffle_profile_softness=2.0e-2,
                phimin=0.0,
                jit_booz=bool(args.jit_booz),
            ),
        )
        qi_times.append(wall_s)

    summary = {
        "runtime": _runtime_info(),
        "input": str(input_path),
        "rebuilt_input": str(rebuilt_input),
        "solver_device": solver_device,
        "vmec_resolution": {"mpol": int(args.mpol), "ntor": int(args.ntor)},
        "qi_resolution": {
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "nphi": int(args.nphi),
            "nalpha": int(args.nalpha),
            "n_bounce": int(args.n_bounce),
            "surfaces": surfaces,
            "jit_booz": bool(args.jit_booz),
        },
        "wall_time_s": {
            "vmec_solve": float(vmec_wall_s),
            "qi_evaluations": [float(value) for value in qi_times],
            "qi_first": float(qi_times[0]),
            "qi_warm_min": float(min(qi_times[1:])) if len(qi_times) > 1 else None,
        },
        "qi_total": None if qi_last is None else float(np.asarray(qi_last["total"])),
        "booz_modes": None if qi_last is None else int(np.asarray(qi_last["booz"]["bmnc_b"]).shape[-1]),
    }
    output_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

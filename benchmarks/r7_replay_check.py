"""Fresh-process replay of a native checkpoint under each evaluation mode.

Reports eta for (a) the checkpoint's declared (base, coordinates) pair in its
saved mode, (b) the same pair under the other mode, and (c) the exported
float64 sum re-anchored as a new base with zero coordinates.  (c) is the
representation-limited stationarity of an ordinary float64 coefficient file.
Read-only diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polish_recovery_sparse as prs  # noqa: E402


def _eta(base, plan, layout, gauge, scale, coordinates):
    force, _ = prs._linear_problem(base, plan, layout, gauge, scale, coordinates)
    constraint = prs.native_tangential_gauge_matrix(base, layout, gauge, scale)
    residual = np.asarray(force(coordinates))
    gradient = np.asarray(jax.jit(jax.grad(lambda c: 0.5 * jnp.vdot(force(c), force(c))))(coordinates))
    projected, projection = prs._sparse_projected_gradient(gradient, constraint)
    normal, _, _ = prs._local_normal_system(force, coordinates, residual, base, plan, layout, gauge, scale)
    frobenius = float(np.sqrt(normal.diagonal().sum()))
    return {
        "residual_norm": float(np.linalg.norm(residual)),
        "projected_gradient_norm": float(np.linalg.norm(projected)),
        "operator_frobenius": frobenius,
        "eta": float(np.linalg.norm(projected)) / (frobenius * float(np.linalg.norm(residual))),
        "projection_true_residual_relative": projection["projection_true_residual_relative"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "schema": "vmex-r7-replay-check/1",
        "state": str(args.state),
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modes": {},
    }
    for stable in (True, False):
        base, accepted, plan, layout, gauge, scale, coordinates, order = prs._load_original_problem(
            args.state, stable_derivatives=stable
        )
        report["modes"][prs.EVALUATION_MODES[int(stable)] + " pair"] = _eta(base, plan, layout, gauge, scale, coordinates)
        # Re-anchor the exported float64 state on the same quadrature; keep
        # the saved metric so the eta normalization is comparable.
        report["modes"][prs.EVALUATION_MODES[int(stable)] + " exported float64 sum"] = _eta(
            accepted, plan, layout, prs.make_native_gauge_plan(accepted, plan), scale, jnp.zeros_like(coordinates)
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

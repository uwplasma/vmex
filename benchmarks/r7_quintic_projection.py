"""Project a certified cubic native state onto a compact quintic basis (R7.4).

A cubic C2 spline is not contained in a quintic C4 simple-knot space, so this
is a declared approximation, not an exact transfer.  Each of the nine radial
channels is fitted in its regular q(s) coordinate by weighted least squares on
Gauss nodes of the target spans, with both clamped end coefficients fixed (the
axis value and the physical edge are preserved exactly).  The value, first,
and second s-derivative errors are recorded against the source spline.  The
output is a fresh checkpoint (initial = accepted, zero coordinates) for the
unchanged sparse driver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polish_recovery_p3 import _checkpoint_arrays, _write_npz_atomic  # noqa: E402
from polish_recovery_refine import _load_state  # noqa: E402

from vmex.core.polish import make_native_correction_layout  # noqa: E402
from vmex.core.polish_variational import (  # noqa: E402
    make_native_gauge_plan,
    make_variational_plan,
    native_coordinate_scales,
)
from vmex.core.radial_basis import BSplineBasis, _span_quadrature  # noqa: E402

CHANNELS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin", "phipf", "chipf", "pressure")


def _project(source: BSplineBasis, target: BSplineBasis, coefficients: np.ndarray, s, w, m=None):
    """Fit each row in physical amplitude rho**m q(s), holding the edge fixed.

    High-m rows store q = a / rho**m, whose coefficients can be ~1e16 near the
    axis; fitting q directly would weight that invisible region.  The axis
    coefficient is held only for m=0 (for m>0 the amplitude vanishes there).
    """

    rows = np.atleast_2d(np.asarray(coefficients, dtype=float))
    m = np.zeros(rows.shape[0], dtype=int) if m is None else np.abs(np.asarray(m, dtype=int))
    rho = np.sqrt(s)
    source_matrix = np.asarray(source.basis_matrix(s))
    target_matrix = np.asarray(target.basis_matrix(s))
    fitted = np.zeros((rows.shape[0], target.size))
    for row, (values, mode) in enumerate(zip(rows, m)):
        power = rho**mode
        samples = power * (source_matrix @ values)
        matrix = power[:, None] * target_matrix
        fixed = [0, target.size - 1] if mode == 0 else [target.size - 1]
        free = [index for index in range(target.size) if index not in fixed]
        # Clamped end coefficients are the end values: source first/last.
        fitted[row, fixed] = values[[0, -1]] if mode == 0 else values[[-1]]
        rhs = samples - matrix[:, fixed] @ fitted[row, fixed]
        root = np.sqrt(w)
        column = np.linalg.norm(root[:, None] * matrix[:, free], axis=0)
        solution, *_ = np.linalg.lstsq(root[:, None] * matrix[:, free] / column, root * rhs, rcond=None)
        fitted[row, free] = solution / column
    return fitted.reshape(np.shape(coefficients)[:-1] + (target.size,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-state", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2, help="keep every stride-th source breakpoint")
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    state = _load_state(args.input_state, prefix="accepted")
    breaks = np.asarray(state.radial_basis.breakpoints)
    kept = breaks[:: args.stride]
    if kept[-1] != breaks[-1]:
        kept = np.append(kept, breaks[-1])
    target = BSplineBasis.clamped(kept, degree=args.degree)
    s, w = _span_quadrature(np.unique(np.concatenate((kept, breaks))), 12)
    modes = np.asarray(state.m)
    projected = {
        name: jnp.asarray(
            _project(state.radial_basis, target, np.asarray(getattr(state, name)), s, w, modes if name[1] == "_" else None)
        )
        for name in CHANNELS
    }
    new = replace(state, radial_basis=target, **projected)
    errors = {}
    for name in CHANNELS:
        old_c, new_c = np.atleast_2d(np.asarray(getattr(state, name))), np.atleast_2d(np.asarray(projected[name]))
        power = np.sqrt(s)[None, :] ** (np.abs(modes)[:, None] if name[1] == "_" else 0)
        entry = {}
        for order in (0, 1, 2):
            # Physical-amplitude error; derivative terms use the q(s) jets
            # weighted by rho**m (the dominant regular factor).
            a = power * (old_c @ np.asarray(state.radial_basis.basis_matrix(s, derivative=order)).T)
            b = power * (new_c @ np.asarray(target.basis_matrix(s, derivative=order)).T)
            entry[f"d{order}_max_abs_error"] = float(np.max(np.abs(a - b)))
            entry[f"d{order}_max_abs_value"] = float(np.max(np.abs(a)))
        errors[name] = entry
    # Same angular grid and radial order as the source solve chart.
    with np.load(args.input_state) as data:
        radial_order = int(data["solve_radial_order"])
        ntheta, nzeta = int(data["solve_ntheta"]), int(data["solve_nzeta"])
    plan = make_variational_plan(new, radial_order=radial_order, ntheta=ntheta, nzeta=nzeta)
    layout = make_native_correction_layout(new)
    gauge = make_native_gauge_plan(new, plan)
    scale = native_coordinate_scales(new, layout, plan)
    arrays = _checkpoint_arrays(new, new, np.zeros(layout.size), scale, gauge)
    arrays["evaluation_mode"] = np.asarray("coefficient-first-split-jets")
    _write_npz_atomic(args.output_state, arrays)
    record = {
        "schema": "vmex-r7-quintic-projection/1",
        "input_state": str(args.input_state),
        "input_state_sha256": hashlib.sha256(args.input_state.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "output_state_sha256": hashlib.sha256(args.output_state.read_bytes()).hexdigest(),
        "exact_transfer": False,
        "source_basis": {"degree": int(state.radial_basis.degree), "size": int(state.radial_basis.size)},
        "target_basis": {"degree": args.degree, "size": int(target.size), "spans": int(kept.size - 1)},
        "coordinates": {"source": int(make_native_correction_layout(state).size), "target": int(layout.size)},
        "q_space_projection_errors": errors,
    }
    args.output_json.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("source_basis", "target_basis", "coordinates")}))
    for name, entry in errors.items():
        print(name, {k: f"{v:.2e}" for k, v in entry.items() if "error" in k})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Scan a local QI objective landscape around a fixed-boundary optimization state.

The scan varies one or two active boundary DOFs as increments around an input
deck, solves VMEC at each point, evaluates existing QI diagnostics, and writes a
JSON/CSV report plus a line or contour-line plot.  It is intentionally a
diagnostic tool, not a new optimizer path.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax._compat import enable_x64
from vmec_jax.optimization import boundary_param_names
from vmec_jax.optimization_workflow import (
    AspectRatio,
    build_fixed_boundary_objective_stage,
    rebuild_for_optimization_resolution,
)
from vmec_jax.quasi_isodynamic.diagnostics import QIDiagnosticOptions, qi_diagnostics_from_state


enable_x64(True)

DEFAULT_OUTPUT_DIR = Path("results/diagnostics/qi_landscape_scan")
DEFAULT_SURFACES = (0.35, 0.65)
METRICS = (
    ("qi_smooth_total", "QI residual"),
    ("qi_mirror_ratio_max", "Mirror ratio"),
    ("qi_max_elongation", "Max elongation"),
    ("aspect", "Aspect"),
    ("mean_iota", "Mean iota"),
)


@dataclass(frozen=True)
class ScanAxis:
    """One scanned boundary parameter and its increment values."""

    dof: str
    values: tuple[float, ...]


def parse_surfaces(raw: str) -> tuple[float, ...]:
    surfaces = tuple(float(part) for part in raw.split(",") if part.strip())
    if not surfaces:
        raise argparse.ArgumentTypeError("--surfaces must contain at least one value")
    for surface in surfaces:
        if surface <= 0.0 or surface > 1.0:
            raise argparse.ArgumentTypeError("QI landscape surfaces must be in (0, 1]")
    return surfaces


def parse_dofs(raw: str) -> tuple[str, ...]:
    dofs = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not 1 <= len(dofs) <= 2:
        raise argparse.ArgumentTypeError("--dofs must contain one or two comma-separated boundary DOF names")
    return dofs


def resolve_input_path(path: Path) -> Path:
    """Resolve either an input file or an optimization output directory."""

    path = Path(path).expanduser()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    candidates = [
        path / "input.final",
        *sorted(path.glob("stage_*_mode*/input.final"), reverse=True),
        path / "input.initial",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No input.final/input.initial file found below {path}")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    out = float(arr.ravel()[0])
    return out if np.isfinite(out) else None


def _base_param_values(stage) -> np.ndarray:
    base_vector = getattr(stage.optimizer, "_base_params_vector", None)
    if callable(base_vector):
        return np.asarray(base_vector(), dtype=float).reshape(-1)
    return np.zeros(len(stage.specs), dtype=float)


def choose_default_dofs(stage, *, count: int) -> tuple[str, ...]:
    """Choose high-amplitude free coefficients when the CLI does not specify DOFs."""

    names = boundary_param_names(stage.specs)
    if not names:
        raise ValueError("No active boundary DOFs are available for this stage.")
    base = _base_param_values(stage)
    order = np.argsort(-np.abs(base)) if base.size == len(names) else np.arange(len(names))
    selected: list[str] = []
    for idx in order:
        name = names[int(idx)]
        if name.lower() == "rc00":
            continue
        selected.append(name)
        if len(selected) == int(count):
            break
    if len(selected) < int(count):
        raise ValueError(f"Requested {count} DOFs but only found {len(selected)} active DOFs.")
    return tuple(selected)


def axis_from_span(dof: str, *, span: float, points: int) -> ScanAxis:
    if int(points) < 2:
        raise ValueError("points must be at least 2")
    return ScanAxis(dof=dof, values=tuple(float(v) for v in np.linspace(-float(span), float(span), int(points))))


def scan_landscape_records(
    *,
    axes: Sequence[ScanAxis],
    specs: Sequence[Any],
    evaluate: Callable[[np.ndarray], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate a 1D or 2D scan using an injected point evaluator."""

    if not 1 <= len(axes) <= 2:
        raise ValueError("Landscape scans support one or two axes.")
    names = boundary_param_names(specs)
    name_to_index = {name: idx for idx, name in enumerate(names)}
    missing = [axis.dof for axis in axes if axis.dof not in name_to_index]
    if missing:
        raise ValueError(f"Unknown boundary DOF(s): {', '.join(missing)}. Available DOFs: {', '.join(names)}")

    records: list[dict[str, Any]] = []
    for point in itertools.product(*(axis.values for axis in axes)):
        params = np.zeros(len(specs), dtype=float)
        deltas = {}
        for axis, value in zip(axes, point):
            params[name_to_index[axis.dof]] = float(value)
            deltas[axis.dof] = float(value)
        diagnostic = dict(evaluate(params))
        metric_values = {key: _finite_float(diagnostic.get(key)) for key, _label in METRICS}
        records.append(
            {
                "deltas": deltas,
                "params": params.tolist(),
                "metrics": metric_values,
                "diagnostics": diagnostic,
            }
        )
    return records


def build_stage(
    *,
    input_path: Path,
    max_mode: int,
    min_vmec_mode: int,
    include: Sequence[str],
    fix: Sequence[str],
    project_input_boundary_to_max_mode: bool,
    inner_max_iter: int,
    inner_ftol: float,
    trial_max_iter: int,
    trial_ftol: float,
    solver_device: str | None,
):
    """Build a fixed-boundary stage used only for forward solves."""

    from vmec_jax import load_config
    from vmec_jax.config import config_from_indata

    cfg, indata = load_config(str(input_path))
    indata = rebuild_for_optimization_resolution(
        indata,
        max_mode=int(max_mode),
        min_vmec_mode=int(min_vmec_mode),
    )
    cfg = config_from_indata(indata)
    objective = AspectRatio().to_objective_term(target=1.0, residual_weight=1.0)
    return build_fixed_boundary_objective_stage(
        cfg,
        indata,
        stage_mode=int(max_mode),
        objectives=[objective],
        include=tuple(include),
        fix=tuple(fix),
        project_input_boundary_to_max_mode=bool(project_input_boundary_to_max_mode),
        min_coeff=0.0,
        inner_max_iter=int(inner_max_iter),
        inner_ftol=float(inner_ftol),
        trial_max_iter=int(trial_max_iter),
        trial_ftol=float(trial_ftol),
        solver_device=solver_device,
    )


def make_qi_evaluator(
    stage,
    *,
    options: QIDiagnosticOptions,
    exact_solve: bool,
) -> Callable[[np.ndarray], dict[str, Any]]:
    """Create a point evaluator that reuses the stage's existing VMEC/QI APIs."""

    def evaluate(params: np.ndarray) -> dict[str, Any]:
        optimizer = stage.optimizer
        state = (
            optimizer._solve_exact_with_tape(params)
            if bool(exact_solve)
            else optimizer._solve_forward(params, trial=True)
        )
        return qi_diagnostics_from_state(
            state=state,
            static=stage.ctx.static,
            indata=stage.ctx.indata,
            signgs=stage.ctx.signgs,
            surfaces=options.surfaces,
            options=options,
            flux_local=stage.ctx.flux,
            prof_local={"pressure": stage.ctx.pressure},
            pressure_local=stage.ctx.pressure,
        )

    return evaluate


def _report_from_records(
    *,
    input_path: Path,
    stage,
    axes: Sequence[ScanAxis],
    records: Sequence[dict[str, Any]],
    options: QIDiagnosticOptions,
    exact_solve: bool,
) -> dict[str, Any]:
    base = _base_param_values(stage)
    names = boundary_param_names(stage.specs)
    return {
        "kind": "qi_landscape_scan",
        "input": str(input_path),
        "dimension": len(axes),
        "dofs": [axis.dof for axis in axes],
        "axes": [{"dof": axis.dof, "values": list(axis.values)} for axis in axes],
        "active_dofs": names,
        "base_parameter_values": {name: float(value) for name, value in zip(names, base)},
        "metrics": [{"key": key, "label": label} for key, label in METRICS],
        "resolution": {
            "surfaces": list(options.surfaces or ()),
            "mboz": options.mboz,
            "nboz": options.nboz,
            "nphi": int(options.nphi),
            "nalpha": int(options.nalpha),
            "n_bounce": int(options.n_bounce),
            "mirror_ntheta": int(options.mirror_ntheta),
            "mirror_nphi": int(options.mirror_nphi),
            "elongation_ntheta": int(options.elongation_ntheta),
            "elongation_nphi": int(options.elongation_nphi),
            "jit_booz": bool(options.jit_booz),
            "exact_solve": bool(exact_solve),
        },
        "records": list(records),
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dofs = list(report["dofs"])
    fieldnames = [f"delta_{dof}" for dof in dofs] + [key for key, _label in METRICS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in report["records"]:
            row = {f"delta_{dof}": record["deltas"].get(dof) for dof in dofs}
            row.update(record["metrics"])
            writer.writerow(row)


def _metric_grid(report: dict[str, Any], metric_key: str) -> np.ndarray:
    axis0 = report["axes"][0]["values"]
    if report["dimension"] == 1:
        return np.asarray([record["metrics"].get(metric_key, np.nan) for record in report["records"]], dtype=float)
    axis1 = report["axes"][1]["values"]
    grid = np.full((len(axis0), len(axis1)), np.nan, dtype=float)
    index0 = {float(value): idx for idx, value in enumerate(axis0)}
    index1 = {float(value): idx for idx, value in enumerate(axis1)}
    dof0, dof1 = report["dofs"]
    for record in report["records"]:
        i = index0[float(record["deltas"][dof0])]
        j = index1[float(record["deltas"][dof1])]
        grid[i, j] = record["metrics"].get(metric_key, np.nan)
    return grid


def plot_report(report: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.0 * len(METRICS), 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    if report["dimension"] == 1:
        x = np.asarray(report["axes"][0]["values"], dtype=float)
        xlabel = f"delta {report['dofs'][0]}"
        for ax, (metric_key, label) in zip(axes, METRICS):
            y = _metric_grid(report, metric_key)
            ax.plot(x, y, marker="o", linewidth=1.4)
            ax.axvline(0.0, color="0.35", linewidth=0.8, linestyle="--")
            ax.set_xlabel(xlabel)
            ax.set_title(label)
            ax.grid(True, alpha=0.25)
    else:
        x = np.asarray(report["axes"][0]["values"], dtype=float)
        y = np.asarray(report["axes"][1]["values"], dtype=float)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        for ax, (metric_key, label) in zip(axes, METRICS):
            z = _metric_grid(report, metric_key)
            finite = z[np.isfinite(z)]
            if finite.size >= 2 and float(np.nanmax(finite)) > float(np.nanmin(finite)):
                levels = min(8, max(3, finite.size // 2))
                contours = ax.contour(xx, yy, z, levels=levels, linewidths=1.0)
                ax.clabel(contours, inline=True, fontsize=7)
            else:
                text = "unavailable" if finite.size == 0 else f"constant {float(finite[0]):.4g}"
                ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center")
            ax.plot([0.0], [0.0], marker="+", color="black", markersize=7)
            ax.set_xlabel(f"delta {report['dofs'][0]}")
            ax.set_ylabel(f"delta {report['dofs'][1]}")
            ax.set_title(label)
            ax.grid(True, alpha=0.25)

    fig.suptitle("QI landscape scan")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="VMEC input file, or an optimization output directory containing input.final.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", type=Path, default=None, help="JSON output path; defaults below --output-dir.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV output path; defaults below --output-dir.")
    parser.add_argument("--plot", type=Path, default=None, help="PNG/PDF plot path; defaults below --output-dir.")
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--min-vmec-mode", type=int, default=5)
    parser.add_argument("--dofs", type=parse_dofs, default=None, help="One or two DOF names, for example rc11,zs11.")
    parser.add_argument("--dimension", type=int, choices=(1, 2), default=2)
    parser.add_argument("--points", type=int, default=5, help="Number of samples along each axis.")
    parser.add_argument("--span", type=float, default=2.0e-2, help="Symmetric increment span for the first axis.")
    parser.add_argument("--span2", type=float, default=None, help="Symmetric increment span for the second axis.")
    parser.add_argument("--surfaces", type=parse_surfaces, default=DEFAULT_SURFACES)
    parser.add_argument("--mboz", type=int, default=10)
    parser.add_argument("--nboz", type=int, default=10)
    parser.add_argument("--nphi", type=int, default=51)
    parser.add_argument("--nalpha", type=int, default=11)
    parser.add_argument("--n-bounce", type=int, default=15)
    parser.add_argument("--include-bounce-endpoints", action="store_true")
    parser.add_argument("--mirror-threshold", type=float, default=0.21)
    parser.add_argument("--mirror-ntheta", type=int, default=48)
    parser.add_argument("--mirror-nphi", type=int, default=48)
    parser.add_argument("--mirror-surface-index", type=int, default=None)
    parser.add_argument("--elongation-threshold", type=float, default=8.0)
    parser.add_argument("--elongation-ntheta", type=int, default=32)
    parser.add_argument("--elongation-nphi", type=int, default=12)
    parser.add_argument("--phimin", type=float, default=0.0)
    parser.add_argument("--no-jit-booz", action="store_true")
    parser.add_argument("--exact-solve", action="store_true", help="Use exact accepted-point solves instead of trial solves.")
    parser.add_argument("--inner-max-iter", type=int, default=60)
    parser.add_argument("--trial-max-iter", type=int, default=60)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-8)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-8)
    parser.add_argument("--solver-device", default=None)
    parser.add_argument("--include", default="rc,zs", help="Boundary coefficient families to include.")
    parser.add_argument("--fix", default="rc00", help="Boundary DOFs to keep fixed.")
    parser.add_argument("--project-input-boundary-to-max-mode", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir)
    json_path = args.json or output_dir / "qi_landscape_scan.json"
    csv_path = args.csv or output_dir / "qi_landscape_scan.csv"
    plot_path = args.plot or output_dir / "qi_landscape_scan.png"

    stage = build_stage(
        input_path=input_path,
        max_mode=args.max_mode,
        min_vmec_mode=args.min_vmec_mode,
        include=tuple(part.strip() for part in args.include.split(",") if part.strip()),
        fix=tuple(part.strip() for part in args.fix.split(",") if part.strip()),
        project_input_boundary_to_max_mode=args.project_input_boundary_to_max_mode,
        inner_max_iter=args.inner_max_iter,
        inner_ftol=args.inner_ftol,
        trial_max_iter=args.trial_max_iter,
        trial_ftol=args.trial_ftol,
        solver_device=args.solver_device,
    )
    requested_dofs = args.dofs or choose_default_dofs(stage, count=int(args.dimension))
    if len(requested_dofs) != int(args.dimension):
        parser.error("--dimension must match the number of DOFs supplied by --dofs")

    spans = [float(args.span), float(args.span if args.span2 is None else args.span2)]
    axes = [
        axis_from_span(dof, span=spans[idx], points=int(args.points))
        for idx, dof in enumerate(requested_dofs)
    ]
    options = QIDiagnosticOptions(
        surfaces=tuple(args.surfaces),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        nphi=int(args.nphi),
        nalpha=int(args.nalpha),
        n_bounce=int(args.n_bounce),
        include_bounce_endpoints=bool(args.include_bounce_endpoints),
        phimin=float(args.phimin),
        jit_booz=not bool(args.no_jit_booz),
        mirror_threshold=float(args.mirror_threshold),
        mirror_ntheta=int(args.mirror_ntheta),
        mirror_nphi=int(args.mirror_nphi),
        mirror_surface_index=args.mirror_surface_index,
        elongation_threshold=float(args.elongation_threshold),
        elongation_ntheta=int(args.elongation_ntheta),
        elongation_nphi=int(args.elongation_nphi),
    )
    records = scan_landscape_records(
        axes=axes,
        specs=stage.specs,
        evaluate=make_qi_evaluator(stage, options=options, exact_solve=bool(args.exact_solve)),
    )
    report = _report_from_records(
        input_path=input_path,
        stage=stage,
        axes=axes,
        records=records,
        options=options,
        exact_solve=bool(args.exact_solve),
    )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_csv(report, csv_path)
    plot_report(report, plot_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

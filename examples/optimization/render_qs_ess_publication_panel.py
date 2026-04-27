#!/usr/bin/env python
"""Render publication-style figures for the QA/QH/QP/QI policy sweep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np

from vmec_jax.plotting import fix_matplotlib_3d, vmecplot2_bmag_grid, vmecplot2_lcfs_3d_grid
from vmec_jax.wout import read_wout


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "results" / "qs_ess_sweep"
REPO_ROOT = SCRIPT_DIR.parents[1]
PROBLEMS = ("qa", "qh", "qp", "qi")
ESS_OPTIONS = (False, True)
POLICIES = ("continuation", "direct")
ROW_SPECS = (
    ("qa", "continuation"),
    ("qa", "direct"),
    ("qh", "continuation"),
    ("qh", "direct"),
    ("qp", "continuation"),
    ("qp", "direct"),
    ("qi", "continuation"),
    ("qi", "direct"),
)
MODES_BY_POLICY = {
    "continuation": (1, 2, 3),
    "direct": (1, 2, 3),
}
_TIMEOUT_SECONDS_RE = re.compile(r"timed out after\s+([0-9]+(?:\.[0-9]+)?)\s*s")


@dataclass(frozen=True)
class CaseResult:
    problem: str
    max_mode: int
    use_ess: bool
    success: bool
    crashed: bool
    message: str
    backend: str = "cpu"
    policy: str = "continuation"
    objective_final: float | None = None
    qs_final: float | None = None
    aspect_final: float | None = None
    iota_final: float | None = None
    nfev: int | None = None
    njev: int | None = None
    total_wall_time_s: float | None = None
    output_dir: str | None = None
    jax_backend: str | None = None
    jax_device_kind: str | None = None
    solver_device: str | None = None
    jax_platforms: str | None = None


@dataclass
class PlotPayload:
    result: CaseResult
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    B_surface: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray
    B_contour: np.ndarray
    nfp: int


def _style_publication():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.17,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    return plt


def _ess_label(use_ess: bool) -> str:
    return "ESS" if use_ess else "No ESS"


def _panel_label(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


def _policy_label(policy: str) -> str:
    return "Continuation" if policy == "continuation" else "Direct start"


def _format_optional_float(value: float | None, fmt: str, *, missing: str = "-") -> str:
    if value is None:
        return missing
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return missing


def _format_wall_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 60.0:.1f}"


def _is_zero_iota_direct_limit(result: CaseResult) -> bool:
    return (
        result.problem == "qa"
        and result.policy == "direct"
        and int(result.max_mode) >= 3
        and result.iota_final is not None
        and abs(float(result.iota_final)) < 5.0e-2
        and result.objective_final is not None
        and float(result.objective_final) > 1.0e-2
    )


def _status_label(result: CaseResult) -> str:
    if result.crashed:
        return "failed"
    if _is_zero_iota_direct_limit(result):
        return "zero-iota"
    return "ok" if result.success else "stopped"


def _row_label(problem: str, policy: str, backend: str | None = None) -> str:
    prefix = "" if backend is None else f"{backend.upper()} | "
    return f"{prefix}{problem.upper()} {_policy_label(policy)}"


def _result_key(result: CaseResult) -> tuple[str, str, str, int, bool]:
    return (result.backend, result.policy, result.problem, int(result.max_mode), bool(result.use_ess))


def _relative_result_parts(path: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(OUTPUT_ROOT).parts
    except ValueError:
        return path.parts


def _infer_backend(path: Path, record: dict) -> tuple[str, bool, bool]:
    if record.get("backend"):
        return str(record["backend"]), True, True
    parts = tuple(part.lower() for part in _relative_result_parts(path))
    for backend in ("cpu", "gpu"):
        if backend in parts:
            return backend, False, True
    return "cpu", False, False


def _infer_policy(path: Path, record: dict) -> tuple[str, bool]:
    if record.get("policy"):
        return str(record["policy"]), True
    parts = tuple(part.lower() for part in _relative_result_parts(path))
    return ("direct", False) if "direct" in parts else ("continuation", False)


def _discovery_priority(path: Path, raw_record: dict) -> int:
    _backend, backend_explicit, backend_in_path = _infer_backend(path, raw_record)
    _policy, policy_explicit = _infer_policy(path, raw_record)
    priority = 0
    if backend_in_path:
        priority += 10
    if policy_explicit:
        priority += 2
    if backend_explicit:
        priority += 20
    return priority


def _infer_missing_wall_time(record: dict) -> None:
    if record.get("total_wall_time_s") is not None:
        return
    match = _TIMEOUT_SECONDS_RE.search(str(record.get("message", "")))
    if match is None:
        return
    record["total_wall_time_s"] = float(match.group(1))


def _lcfs_xyz(R: np.ndarray, Z: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = R * np.cos(phi[None, :])
    Y = R * np.sin(phi[None, :])
    return X, Y, Z


def _pi_label(v: float) -> str:
    from fractions import Fraction

    if abs(v) < 1e-14:
        return "0"
    frac = Fraction(v / float(np.pi)).limit_denominator(64)
    n, d = frac.numerator, frac.denominator
    if d == 1:
        return "pi" if n == 1 else f"{n}pi"
    return f"pi/{d}" if n == 1 else f"{n}pi/{d}"


def _discover_results() -> list[CaseResult]:
    results_by_key: dict[tuple[str, str, str, int, bool], tuple[int, float, CaseResult]] = {}
    for path in sorted(OUTPUT_ROOT.glob("**/case_result.json")):
        raw_record = json.loads(path.read_text())
        priority = _discovery_priority(path, raw_record)
        record = dict(raw_record)
        backend, backend_explicit, backend_in_path = _infer_backend(path, raw_record)
        if not backend_explicit and not backend_in_path:
            # Ignore pre-backend-layout results. They caused stale timeout/zero-iota
            # records to leak into regenerated CPU/GPU panels after new sweeps.
            continue
        policy, _policy_explicit = _infer_policy(path, raw_record)
        record["backend"] = backend
        record["policy"] = policy
        _infer_missing_wall_time(record)
        output_dir = record.get("output_dir")
        if output_dir is None or not Path(str(output_dir)).exists():
            record["output_dir"] = str(path.parent)
        result = CaseResult(**record)
        key = _result_key(result)
        mtime = path.stat().st_mtime
        previous = results_by_key.get(key)
        if previous is None or (priority, mtime) >= (previous[0], previous[1]):
            results_by_key[key] = (priority, mtime, result)
    results = [result for _priority, _mtime, result in results_by_key.values()]
    if not results:
        raise FileNotFoundError(f"No case_result.json files found under {OUTPUT_ROOT}")
    return results


def _write_combined_summary(results: list[CaseResult]) -> None:
    ordered = sorted(results, key=lambda r: (r.backend, POLICIES.index(r.policy), r.problem, r.max_mode, r.use_ess))
    records = [_summary_record(r) for r in ordered]
    (OUTPUT_ROOT / "summary_all.json").write_text(json.dumps(records, indent=2))
    with (OUTPUT_ROOT / "summary_all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "policy",
                "backend",
                "problem",
                "max_mode",
                "use_ess",
                "success",
                "crashed",
                "objective_final",
                "qs_final",
                "aspect_final",
                "iota_final",
                "nfev",
                "njev",
                "total_wall_time_s",
                "jax_backend",
                "jax_device_kind",
                "solver_device",
                "jax_platforms",
                "message",
                "output_dir",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _summary_record(result: CaseResult) -> dict:
    record = asdict(result)
    output_dir = record.get("output_dir")
    if output_dir:
        try:
            record["output_dir"] = str(Path(str(output_dir)).resolve().relative_to(REPO_ROOT))
        except ValueError:
            record["output_dir"] = str(output_dir)
    return record


def _result_lookup(results: list[CaseResult]) -> dict[tuple[str, str, str, int, bool], CaseResult]:
    return {_result_key(result): result for result in results}


def _draw_placeholder(ax, message: str, *, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f6f7f9")
    text_kwargs = {
        "transform": ax.transAxes,
        "ha": "center",
        "va": "center",
        "fontsize": 10,
        "color": "0.42",
        "style": "italic",
    }
    if hasattr(ax, "text2D"):
        ax.text2D(0.5, 0.5, message, **text_kwargs)
    else:
        ax.text(0.5, 0.5, message, **text_kwargs)
    if title is not None:
        ax.set_title(title, fontsize=11)


def _history_for(result: CaseResult) -> dict | None:
    if result.output_dir is None:
        return None
    path = Path(result.output_dir) / "history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _lookup_result(
    lookup: dict[tuple[str, str, str, int, bool], CaseResult],
    *,
    backend: str,
    problem: str,
    max_mode: int,
    use_ess: bool,
    policy: str,
    allow_mode1_baseline: bool = False,
) -> CaseResult | None:
    result = lookup.get((backend, policy, problem, max_mode, use_ess))
    if result is not None:
        return result
    if allow_mode1_baseline and policy == "direct" and max_mode == 1:
        return lookup.get((backend, "continuation", problem, 1, use_ess))
    return None


def _row_has_history(
    lookup: dict[tuple[str, str, str, int, bool], CaseResult],
    *,
    backend: str,
    problem: str,
    policy: str,
) -> bool:
    for max_mode in MODES_BY_POLICY[policy]:
        for use_ess in ESS_OPTIONS:
            result = _lookup_result(
                lookup,
                backend=backend,
                problem=problem,
                max_mode=max_mode,
                use_ess=use_ess,
                policy=policy,
                allow_mode1_baseline=True,
            )
            if result is not None:
                return True
    return False


def _available_row_specs(results: list[CaseResult]) -> list[tuple[str, str, str]]:
    lookup = _result_lookup(results)
    backends = sorted({result.backend for result in results})
    return [
        (backend, problem, policy)
        for backend in backends
        for problem, policy in ROW_SPECS
        if _row_has_history(lookup, backend=backend, problem=problem, policy=policy)
    ]


def _problems_with_payloads(
    payloads: dict[tuple[str, str, str, int, bool], PlotPayload],
    *,
    backend: str,
    policy: str,
) -> tuple[str, ...]:
    modes = MODES_BY_POLICY[policy]
    return tuple(
        problem
        for problem in PROBLEMS
        if any((backend, policy, problem, mode, use_ess) in payloads for mode in modes for use_ess in ESS_OPTIONS)
    )


def _load_payloads(results: list[CaseResult]) -> dict[tuple[str, str, str, int, bool], PlotPayload]:
    payloads: dict[tuple[str, str, str, int, bool], PlotPayload] = {}
    for result in results:
        if result.crashed or result.output_dir is None:
            continue
        wout_path = Path(result.output_dir) / "wout_final.nc"
        if not wout_path.exists():
            continue
        wout = read_wout(str(wout_path))
        ns = int(np.asarray(wout.ns))
        theta3d, phi3d, R3d, Z3d, B3d = vmecplot2_lcfs_3d_grid(
            wout,
            s_index=ns - 1,
            ntheta=48,
            nzeta=96,
        )
        X3d, Y3d, Z3d = _lcfs_xyz(R3d, Z3d, phi3d)
        zeta_max = 2.0 * np.pi / int(np.asarray(wout.nfp))
        theta2d, zeta2d, B2d = vmecplot2_bmag_grid(
            wout,
            s_index=ns - 1,
            ntheta=128,
            nzeta=192,
            zeta_max=zeta_max,
        )
        payloads[_result_key(result)] = PlotPayload(
            result=result,
            X=X3d,
            Y=Y3d,
            Z=Z3d,
            B_surface=B3d,
            theta=theta2d,
            zeta=zeta2d,
            B_contour=B2d,
            nfp=int(np.asarray(wout.nfp)),
        )
    return payloads


def _plot_objective_panel_all_policies(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()

    lookup = _result_lookup(results)
    row_specs = _available_row_specs(results)
    if not row_specs:
        raise ValueError("No optimization histories are available to plot")
    fig, axes = plt.subplots(len(row_specs), 3, figsize=(18.2, 3.65 * len(row_specs)), sharey="row")
    if len(row_specs) == 1:
        axes = np.asarray([axes])
    colors = {False: "#1f77b4", True: "#d95f02"}
    line_labels = {False: "No ESS", True: "ESS"}
    mode_titles = ("Mode 1 baseline", "Mode 2", "Mode 3")

    for row_index, (backend, problem, policy) in enumerate(row_specs):
        for col_index, max_mode in enumerate((1, 2, 3)):
            ax = axes[row_index, col_index]
            panel_label = _panel_label(row_index * 3 + col_index)
            ax.text(
                0.01,
                0.99,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
            )
            if row_index == 0:
                ax.set_title(mode_titles[col_index], fontsize=12)
            annotation_lines = []
            plotted_any = False
            baseline_note = policy == "direct" and max_mode == 1
            for use_ess in ESS_OPTIONS:
                result = _lookup_result(
                    lookup,
                    backend=backend,
                    problem=problem,
                    max_mode=max_mode,
                    use_ess=use_ess,
                    policy=policy,
                    allow_mode1_baseline=True,
                )
                if result is None:
                    continue
                history = _history_for(result)
                if history is None:
                    continue
                objective = np.asarray([max(float(entry["objective"]), 1e-16) for entry in history["history"]], dtype=float)
                x = np.arange(len(objective), dtype=float)
                linestyle = "-" if result.success and not result.crashed else "--"
                alpha = 0.70 if baseline_note else 1.0
                linewidth = 2.5 if use_ess else 2.1
                label = line_labels[use_ess] if (row_index == 0 and col_index == 0) else None
                ax.semilogy(
                    x,
                    objective,
                    color=colors[use_ess],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )
                plotted_any = True
                ax.scatter(x[-1], objective[-1], color=colors[use_ess], s=28, zorder=4, alpha=alpha)
                for boundary in history.get("stage_boundaries", [])[:-1]:
                    ax.axvline(float(boundary), color="0.75", linestyle=":", linewidth=1.0, zorder=0)

                wall_min = float(result.total_wall_time_s) / 60.0
                meta = f"{line_labels[use_ess]}: J={float(result.objective_final):.2e}, {wall_min:.1f}m"
                meta += f", A={float(result.aspect_final):.3f}"
                if result.iota_final is not None:
                    meta += f", iota={float(result.iota_final):.4f}"
                if result.solver_device:
                    meta += f", dev={result.solver_device}"
                if _is_zero_iota_direct_limit(result):
                    meta += ", zero-iota branch"
                annotation_lines.append(meta)

            if baseline_note:
                ax.text(
                    0.98,
                    0.98,
                    "shared mode-1 baseline",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color="0.35",
                )
            row_title = _row_label(problem, policy, backend)
            if col_index == 0:
                ax.set_ylabel(f"{row_title}\nTotal objective", fontsize=11)
            if row_index == len(row_specs) - 1:
                ax.set_xlabel("History index", fontsize=11)
            ax.set_xlim(left=0)
            ax.grid(True, which="both", alpha=0.20)
            if not plotted_any:
                fallback_results = [
                    _lookup_result(
                        lookup,
                        backend=backend,
                        problem=problem,
                        max_mode=max_mode,
                        use_ess=use_ess,
                        policy=policy,
                        allow_mode1_baseline=True,
                    )
                    for use_ess in ESS_OPTIONS
                ]
                fallback_lines = []
                for use_ess, result in zip(ESS_OPTIONS, fallback_results, strict=True):
                    if result is None:
                        continue
                    if result.crashed:
                        status = "timed out" if "timed out" in result.message.lower() else "failed"
                    else:
                        status = "no history"
                    wall = _format_wall_minutes(result.total_wall_time_s)
                    if wall != "-":
                        status = f"{status}, {wall} min"
                    fallback_lines.append(f"{line_labels[use_ess]}: {status}")
                if fallback_lines:
                    placeholder = "\n".join(fallback_lines)
                else:
                    placeholder = "pending"
                _draw_placeholder(ax, placeholder)
                ax.text(
                    0.01,
                    0.99,
                    panel_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=13,
                    fontweight="bold",
                )
            if annotation_lines:
                ax.text(
                    0.02,
                    0.02,
                    "\n".join(annotation_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.3,
                    bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "0.86", "alpha": 0.92},
                )

    from matplotlib.lines import Line2D

    handles, labels = axes[0, 0].get_legend_handles_labels()
    status_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=2.2, label="solid: optimizer success"),
        Line2D([0], [0], color="0.25", linestyle="--", linewidth=2.2, label="dashed: stopped/diagnostic budget"),
    ]
    if handles:
        fig.legend(
            handles + status_handles,
            labels + [handle.get_label() for handle in status_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=4,
            frameon=False,
        )
    fig.suptitle(
        "QA/QH/QP/QI optimization histories by backend: continuation versus direct-start mode expansion",
        y=0.998,
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_state_atlas(
    results: list[CaseResult],
    payloads: dict[tuple[str, str, str, int, bool], PlotPayload],
    *,
    policy: str,
    backend: str = "cpu",
    outpath_png: Path,
    outpath_pdf: Path,
) -> bool:
    plt = _style_publication()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    lookup = _result_lookup(results)
    modes = MODES_BY_POLICY[policy]
    problems = _problems_with_payloads(payloads, backend=backend, policy=policy)
    if not problems:
        return False
    columns = [(mode, use_ess) for mode in modes for use_ess in ESS_OPTIONS]
    ncols = len(columns)
    nrows = 2 * len(problems)
    fig = plt.figure(figsize=(3.95 * ncols, 3.15 * nrows))
    grid = fig.add_gridspec(nrows, ncols, wspace=0.14, hspace=0.22)

    def _finite_range(values: np.ndarray) -> tuple[float, float]:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 1.0
        if vmax <= vmin:
            pad = max(abs(vmin), 1.0) * 1e-12
            vmin -= pad
            vmax += pad
        return vmin, vmax

    def _inset_colorbar(mappable, ax, *, label: str, height: str) -> None:
        cax = inset_axes(ax, width="4.5%", height=height, loc="lower right", borderpad=0.62)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(label, fontsize=7.2)
        cbar.ax.tick_params(labelsize=6.7, length=2)

    for col_index, (max_mode, use_ess) in enumerate(columns):
        for problem_index, problem in enumerate(problems):
            row_surface = problem_index * 2
            row_contour = row_surface + 1
            result = _lookup_result(
                lookup,
                backend=backend,
                problem=problem,
                max_mode=max_mode,
                use_ess=use_ess,
                policy=policy,
                allow_mode1_baseline=True,
            )
            payload = payloads.get(_result_key(result)) if result is not None else None
            title = None
            if problem_index == 0:
                title = f"mode {max_mode} | {_ess_label(use_ess)}"
            if result is None or payload is None:
                ax3d = fig.add_subplot(grid[row_surface, col_index], projection="3d")
                _draw_placeholder(ax3d, "pending", title=title)
                if col_index == 0:
                    ax3d.text2D(
                        -0.16,
                        0.5,
                        f"{problem.upper()} LCFS",
                        transform=ax3d.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                ax2d = fig.add_subplot(grid[row_contour, col_index])
                _draw_placeholder(ax2d, "pending")
                if col_index == 0:
                    ax2d.set_ylabel("theta")
                    ax2d.text(
                        -0.19,
                        0.5,
                        f"{problem.upper()} |B|",
                        transform=ax2d.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                    )
                ax2d.set_xlabel("zeta")
                continue

            ax3d = fig.add_subplot(grid[row_surface, col_index], projection="3d")
            b3_min, b3_max = _finite_range(payload.B_surface)
            surface_norm = Normalize(vmin=b3_min, vmax=b3_max)
            facecolors = cm.viridis(surface_norm(payload.B_surface))
            ax3d.plot_surface(
                payload.X,
                payload.Y,
                payload.Z,
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                linewidth=0,
                antialiased=False,
                shade=False,
            )
            ax3d.view_init(elev=24, azim=42)
            fix_matplotlib_3d(ax3d)
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            sm = ScalarMappable(norm=surface_norm, cmap=cm.viridis)
            sm.set_array([])
            _inset_colorbar(sm, ax3d, label="|B| (T)", height="46%")
            if problem_index == 0:
                wall_min = float(result.total_wall_time_s) / 60.0
                ax3d.set_title(
                    f"mode {max_mode} | {_ess_label(use_ess)}\nJ={float(result.objective_final):.2e}, {wall_min:.1f} min",
                    pad=10,
                )
            if col_index == 0:
                ax3d.text2D(
                    -0.16,
                    0.5,
                    f"{problem.upper()} LCFS",
                    transform=ax3d.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )

            ax2d = fig.add_subplot(grid[row_contour, col_index])
            zeta_mesh, theta_mesh = np.meshgrid(payload.zeta, payload.theta)
            b2_min, b2_max = _finite_range(payload.B_contour)
            contour_levels = np.linspace(b2_min, b2_max, 22)
            contours = ax2d.contour(
                zeta_mesh,
                theta_mesh,
                payload.B_contour,
                levels=contour_levels,
                cmap="viridis",
                linewidths=1.0,
            )
            _inset_colorbar(contours, ax2d, label="|B| (T)", height="78%")
            ax2d.set_ylim(0.0, 2.0 * np.pi)
            ax2d.set_xlim(0.0, float(np.max(payload.zeta)))
            ax2d.set_yticks([0.0, np.pi, 2.0 * np.pi])
            if col_index == 0:
                ax2d.set_yticklabels(["0", "pi", "2pi"])
                ax2d.set_ylabel("theta")
                ax2d.text(
                    -0.19,
                    0.5,
                    f"{problem.upper()} |B|",
                    transform=ax2d.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax2d.set_yticklabels([])
            xticks = [0.0, float(np.max(payload.zeta)) / 2.0, float(np.max(payload.zeta))]
            ax2d.set_xticks(xticks)
            ax2d.set_xticklabels([_pi_label(v) for v in xticks])
            ax2d.set_xlabel("zeta")
            ax2d.grid(False)
            wall_min = float(result.total_wall_time_s) / 60.0
            meta = f"A={float(result.aspect_final):.3f}\nwall={wall_min:.1f} min"
            if result.iota_final is not None:
                meta = f"A={float(result.aspect_final):.3f}\niota={float(result.iota_final):.4f}\nwall={wall_min:.1f} min"
            if _is_zero_iota_direct_limit(result):
                meta += "\nzero-iota branch"
            ax2d.text(
                0.02,
                0.02,
                meta,
                transform=ax2d.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.1,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.86", "alpha": 0.92},
            )

    fig.suptitle(
        f"Final-state atlas: {backend.upper()} {_policy_label(policy)} policy",
        y=0.995,
        fontsize=16,
    )
    fig.subplots_adjust(left=0.045, right=0.985, top=0.93, bottom=0.05)
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_summary_tables(results: list[CaseResult], outpath_png: Path, outpath_pdf: Path) -> None:
    plt = _style_publication()

    row_specs = _available_row_specs(results)
    if not row_specs:
        raise ValueError("No optimization summary rows are available to plot")
    ncols = 2
    nrows = int(np.ceil(len(row_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5.25 * nrows))
    axes_arr = np.asarray(axes).ravel()
    for ax in axes_arr[len(row_specs):]:
        ax.axis("off")
    for ax, (backend, problem, policy) in zip(axes_arr, row_specs):
        ax.axis("off")
        lookup = _result_lookup(results)
        group = [
            result
            for max_mode in MODES_BY_POLICY[policy]
            for use_ess in ESS_OPTIONS
            if (
                result := _lookup_result(
                    lookup,
                    backend=backend,
                    problem=problem,
                    max_mode=max_mode,
                    use_ess=use_ess,
                    policy=policy,
                    allow_mode1_baseline=True,
                )
            )
            is not None
        ]
        if not group:
            _draw_placeholder(ax, "pending")
            ax.set_title(_row_label(problem, policy, backend), fontsize=12, pad=8)
            continue
        finite_objective_indices = [
            i for i, result in enumerate(group) if result.objective_final is not None and not result.crashed
        ]
        best_index = (
            min(finite_objective_indices, key=lambda i: float(group[i].objective_final))
            if finite_objective_indices
            else -1
        )
        if problem in ("qa", "qp"):
            columns = ["Configuration", "Status", "Final J", "Aspect", "Iota", "nfev", "Wall (min)"]
            widths = [0.27, 0.13, 0.16, 0.11, 0.11, 0.08, 0.14]
            rows = [
                [
                    f"mode {result.max_mode} | {_ess_label(result.use_ess)}",
                    _status_label(result),
                    _format_optional_float(result.objective_final, ".2e"),
                    _format_optional_float(result.aspect_final, ".4f"),
                    _format_optional_float(result.iota_final, ".4f"),
                    "-" if result.nfev is None else f"{int(result.nfev)}",
                    _format_wall_minutes(result.total_wall_time_s),
                ]
                for result in group
            ]
        else:
            columns = ["Configuration", "Status", "Final J", "Aspect", "nfev", "Wall (min)"]
            widths = [0.32, 0.14, 0.18, 0.13, 0.09, 0.14]
            rows = [
                [
                    f"mode {result.max_mode} | {_ess_label(result.use_ess)}",
                    _status_label(result),
                    _format_optional_float(result.objective_final, ".2e"),
                    _format_optional_float(result.aspect_final, ".4f"),
                    "-" if result.nfev is None else f"{int(result.nfev)}",
                    _format_wall_minutes(result.total_wall_time_s),
                ]
                for result in group
            ]
        table = ax.table(
            cellText=rows,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colLoc="center",
            colWidths=widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.4)
        table.scale(1.0, 1.55)
        for (row_index, _col_index), cell in table.get_celld().items():
            if row_index == 0:
                cell.set_facecolor("#e9eef5")
                cell.set_text_props(weight="bold")
            else:
                cell.set_edgecolor("0.82")
                is_ess_row = rows[row_index - 1][0].split("|", 1)[1].strip() == "ESS"
                is_failed_row = rows[row_index - 1][1] == "failed"
                is_limited_row = rows[row_index - 1][1] == "zero-iota"
                if row_index - 1 == best_index:
                    cell.set_facecolor("#e6f4ea")
                elif is_failed_row:
                    cell.set_facecolor("#f8d7da")
                elif is_limited_row:
                    cell.set_facecolor("#fff3cd")
                elif is_ess_row:
                    cell.set_facecolor("#fff3e8")
                else:
                    cell.set_facecolor("#eef5ff")
        ax.set_title(_row_label(problem, policy, backend), fontsize=12, pad=8)

    fig.suptitle("Sweep summary tables", y=0.99, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _assemble_full_panel(
    image_paths: list[Path],
    titles: list[str],
    outpath_png: Path,
    outpath_pdf: Path,
) -> None:
    plt = _style_publication()
    import matplotlib.image as mpimg

    images = [mpimg.imread(path) for path in image_paths]
    if not images:
        raise ValueError("No image paths were provided for the full panel")
    height_ratios = [1.0]
    if len(images) > 2:
        height_ratios.extend([1.05] * (len(images) - 2))
    if len(images) > 1:
        height_ratios.append(0.62)
    fig, axes = plt.subplots(
        len(images),
        1,
        figsize=(22, 9.5 + 8.2 * len(images)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.asarray(axes).ravel()
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=16, pad=10, fontweight="bold")
    fig.suptitle("Reviewer-facing QA/QH/QP/QI optimization policy sweep panel", y=0.996, fontsize=19)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    fig.savefig(outpath_png, dpi=220, bbox_inches="tight")
    fig.savefig(outpath_pdf, bbox_inches="tight")
    plt.close(fig)


def _copy_alias(src_stem: Path, dst_stem: Path) -> None:
    """Copy a png/pdf figure pair so legacy names do not stay stale."""
    for suffix in (".png", ".pdf"):
        src = src_stem.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst_stem.with_suffix(suffix))


def main() -> None:
    results = _discover_results()
    _write_combined_summary(results)
    payloads = _load_payloads(results)

    objective_png = OUTPUT_ROOT / "objective_panel_all_policies.png"
    objective_pdf = OUTPUT_ROOT / "objective_panel_all_policies.pdf"
    summary_png = OUTPUT_ROOT / "summary_tables_all_policies.png"
    summary_pdf = OUTPUT_ROOT / "summary_tables_all_policies.pdf"
    panel_png = OUTPUT_ROOT / "publication_panel_full.png"
    panel_pdf = OUTPUT_ROOT / "publication_panel_full.pdf"

    _plot_objective_panel_all_policies(results, objective_png, objective_pdf)
    _copy_alias(OUTPUT_ROOT / "objective_panel_all_policies", OUTPUT_ROOT / "objective_panel")
    for backend in sorted({result.backend for result in results}):
        backend_results = [result for result in results if result.backend == backend]
        _plot_objective_panel_all_policies(
            backend_results,
            OUTPUT_ROOT / f"objective_panel_{backend}_policies.png",
            OUTPUT_ROOT / f"objective_panel_{backend}_policies.pdf",
        )
    atlas_paths: list[tuple[Path, Path, str]] = []
    for backend in sorted({result.backend for result in results}):
        for policy in POLICIES:
            if not _problems_with_payloads(payloads, backend=backend, policy=policy):
                continue
            atlas_png = OUTPUT_ROOT / f"final_state_atlas_{backend}_{policy}.png"
            atlas_pdf = OUTPUT_ROOT / f"final_state_atlas_{backend}_{policy}.pdf"
            wrote_atlas = _plot_state_atlas(
                results,
                payloads,
                policy=policy,
                backend=backend,
                outpath_png=atlas_png,
                outpath_pdf=atlas_pdf,
            )
            if wrote_atlas:
                atlas_paths.append(
                    (
                        atlas_png,
                        atlas_pdf,
                        f"Final-state atlas: {backend.upper()} {_policy_label(policy).lower()} policy",
                    )
                )
                if backend == "cpu":
                    # Keep the backend-qualified files for CPU/GPU comparisons,
                    # and also publish the concise filenames requested in the
                    # README/docs for the production-accuracy CPU final states.
                    shutil.copy2(atlas_png, OUTPUT_ROOT / f"final_state_atlas_{policy}.png")
                    shutil.copy2(atlas_pdf, OUTPUT_ROOT / f"final_state_atlas_{policy}.pdf")
                    if policy == "continuation":
                        _copy_alias(OUTPUT_ROOT / f"final_state_atlas_{policy}", OUTPUT_ROOT / "geometry_atlas")
    _plot_summary_tables(results, summary_png, summary_pdf)
    _copy_alias(OUTPUT_ROOT / "summary_tables_all_policies", OUTPUT_ROOT / "summary_table")
    for backend in sorted({result.backend for result in results}):
        backend_results = [result for result in results if result.backend == backend]
        _plot_summary_tables(
            backend_results,
            OUTPUT_ROOT / f"summary_tables_{backend}_policies.png",
            OUTPUT_ROOT / f"summary_tables_{backend}_policies.pdf",
        )
    image_paths = [objective_png] + [path for path, _pdf, _title in atlas_paths] + [summary_png]
    panel_titles = ["A. Objective histories: continuation and direct-start policies"]
    for index, (_path, _pdf, title) in enumerate(atlas_paths, start=1):
        panel_titles.append(f"{chr(ord('A') + index)}. {title}")
    panel_titles.append(f"{chr(ord('A') + len(panel_titles))}. Sweep summary tables")
    _assemble_full_panel(
        image_paths,
        panel_titles,
        panel_png,
        panel_pdf,
    )
    _copy_alias(OUTPUT_ROOT / "publication_panel_full", OUTPUT_ROOT / "publication_panel")

    print(f"Wrote {objective_png}")
    for atlas_png, _atlas_pdf, _title in atlas_paths:
        print(f"Wrote {atlas_png}")
    print(f"Wrote {summary_png}")
    print(f"Wrote {panel_png}")


if __name__ == "__main__":
    main()

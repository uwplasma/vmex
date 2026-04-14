"""Generate the README runtime/memory report from benchmark summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mem_bytes(rec: dict[str, Any] | None) -> int | None:
    if rec is None:
        return None
    peak = rec.get("peak_footprint_bytes")
    if isinstance(peak, int) and peak > 0:
        return peak
    rss = rec.get("max_rss_bytes")
    if isinstance(rss, int) and rss > 0:
        return rss
    return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _format_gib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / (1024 ** 3):.2f} GiB"


def _collect_records(
    cpu_summaries: list[dict[str, Any]],
    gpu_summaries: list[dict[str, Any]] | None = None,  # kept for API compat, ignored
) -> list[dict[str, Any]]:
    cpu_cases: dict[str, dict[str, Any]] = {}
    cpu_results: dict[str, dict[str, Any]] = {}
    cpu_vmecpp: dict[str, dict[str, Any]] = {}

    for summary in cpu_summaries:
        for rec in summary.get("cases", []):
            cpu_cases[str(rec["id"])] = dict(rec)
        for rec in summary.get("results", []):
            case_id = str(rec.get("case_id"))
            if rec.get("backend") == "vmec2000":
                cpu_results.setdefault(case_id, {})["vmec2000"] = rec
            elif rec.get("backend") == "vmec_jax":
                cpu_results.setdefault(case_id, {})["cpu"] = rec
            elif rec.get("backend") == "vmecpp":
                cpu_vmecpp[case_id] = rec

    rows: list[dict[str, Any]] = []
    for case_id in sorted(cpu_cases):
        case = cpu_cases[case_id]
        vmec = cpu_results.get(case_id, {}).get("vmec2000")
        cpu = cpu_results.get(case_id, {}).get("cpu")
        pp = cpu_vmecpp.get(case_id)
        vmec_ok = bool(vmec is not None and vmec.get("ok", True))
        cpu_ok = bool(cpu is not None and cpu.get("ok", True))
        pp_ok = bool(pp is not None and pp.get("ok", True))

        vmec_rt = None if (vmec is None or not vmec_ok) else float(vmec.get("runtime_s", vmec.get("time_real_s", np.nan)))
        cpu_rt = None if (cpu is None or not cpu_ok) else float(
            cpu.get("runtime_warm_s", cpu.get("runtime_s", cpu.get("time_real_s", np.nan)))
        )
        pp_rt = None if (pp is None or not pp_ok) else float(pp.get("runtime_s", pp.get("time_real_s", np.nan)))
        vmec_mem = _mem_bytes(vmec) if vmec_ok else None
        cpu_mem = _mem_bytes(cpu) if cpu_ok else None
        pp_mem = _mem_bytes(pp) if pp_ok else None
        rows.append(
            {
                "id": case_id,
                "lfreeb": bool(case.get("lfreeb", False)),
                "lasym": bool(case.get("lasym", False)),
                "axisymmetric": bool(case.get("axisymmetric", False)),
                "vmec2000": vmec,
                "cpu": cpu,
                "vmecpp": pp,
                "vmec_runtime_s": vmec_rt,
                "cpu_runtime_s": cpu_rt,
                "vmecpp_runtime_s": pp_rt,
                "vmec_mem_bytes": vmec_mem,
                "cpu_mem_bytes": cpu_mem,
                "vmecpp_mem_bytes": pp_mem,
            }
        )
    return rows


def _ratio(value: float | int | None, baseline: float | int | None) -> float | None:
    if value is None or baseline is None:
        return None
    if float(baseline) <= 0.0:
        return None
    return float(value) / float(baseline)


def _speedup(value: float | int | None, baseline: float | int | None) -> float | None:
    if value is None or baseline is None:
        return None
    if float(value) <= 0.0:
        return None
    return float(baseline) / float(value)


def _write_markdown_table(rows: list[dict[str, Any]], outpath: Path) -> None:
    include_vmecpp = any(row.get("vmecpp_runtime_s") is not None or row.get("vmecpp_mem_bytes") is not None for row in rows)
    if include_vmecpp:
        header = [
            "| Example | Boundary | Topology | LASYM | VMEC2000 runtime | VMEC2000 memory | vmec_jax runtime (warmed) | vmec_jax memory | VMEC++ runtime | VMEC++ memory |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    else:
        header = [
            "| Example | Boundary | Topology | LASYM | VMEC2000 runtime | VMEC2000 memory | vmec_jax runtime (warmed) | vmec_jax memory |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    lines = list(header)
    for row in rows:
        boundary = "free" if row["lfreeb"] else "fixed"
        topology = "axisym" if row["axisymmetric"] else "non-axisym"
        lasym = "true" if row["lasym"] else "false"
        cols = [
            row["id"],
            boundary,
            topology,
            lasym,
            _format_seconds(row["vmec_runtime_s"]),
            _format_gib(row["vmec_mem_bytes"]),
            _format_seconds(row["cpu_runtime_s"]),
            _format_gib(row["cpu_mem_bytes"]),
        ]
        if include_vmecpp:
            cols.extend([_format_seconds(row.get("vmecpp_runtime_s")), _format_gib(row.get("vmecpp_mem_bytes"))])
        lines.append("| " + " | ".join(cols) + " |")
    outpath.write_text("\n".join(lines) + "\n")


def _draw_speedup_panel(
    ax,
    *,
    rows: list[dict[str, Any]],
    value_key: str,
    base_key: str,
    title: str,
    color: str,
) -> None:
    labels = [row["id"] for row in rows]
    speedup = np.array([_speedup(row[value_key], row[base_key]) or np.nan for row in rows], dtype=float)
    y = np.arange(len(rows), dtype=float)

    ax.barh(y, speedup, height=0.64, color=color)
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("speedup (>1 is faster)")
    ax.grid(axis="x", alpha=0.25, which="both")


def _write_runtime_figure(rows: list[dict[str, Any]], outpath: Path, *, figure_kind: str) -> None:
    if figure_kind == "fixed":
        rows = [row for row in rows if not bool(row["lfreeb"])]
    elif figure_kind == "freeb":
        rows = [row for row in rows if bool(row["lfreeb"])]
    if not rows:
        raise ValueError(f"No rows available for figure_kind={figure_kind!r}.")
    include_vmecpp = any(row.get("vmecpp_runtime_s") is not None for row in rows)
    rows = sorted(
        rows,
        key=lambda row: (
            -max(
                (_speedup(row["cpu_runtime_s"], row["vmec_runtime_s"]) or 0.0),
                (_speedup(row.get("vmecpp_runtime_s"), row["vmec_runtime_s"]) or 0.0),
            ),
            row["id"],
        ),
    )
    labels = [row["id"] for row in rows]
    y = np.arange(len(rows), dtype=float)
    n_series = 2 + int(include_vmecpp)
    height = min(0.22, 0.9 / n_series)
    offsets = np.linspace(-(n_series - 1) / 2 * height, (n_series - 1) / 2 * height, n_series)
    fig, ax = plt.subplots(1, 1, figsize=(14.5, max(8.0, 0.42 * len(rows) + 1.6)))
    vmec = np.array([row["vmec_runtime_s"] if row["vmec_runtime_s"] is not None else np.nan for row in rows], dtype=float)
    cpu = np.array([row["cpu_runtime_s"] if row["cpu_runtime_s"] is not None else np.nan for row in rows], dtype=float)
    series = [("#1f77b4", "VMEC2000", vmec), ("#ff7f0e", "vmec_jax (warmed)", cpu)]
    if include_vmecpp:
        pp = np.array(
            [row.get("vmecpp_runtime_s") if row.get("vmecpp_runtime_s") is not None else np.nan for row in rows], dtype=float
        )
        series.append(("#2ca02c", "VMEC++", pp))
    for (color, label, vals), off in zip(series, offsets):
        ax.barh(y + off, vals, height=height, color=color, label=label)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("runtime (seconds, log scale)")
    ax.grid(axis="x", alpha=0.18, which="both")
    ax.legend(frameon=False, ncol=n_series, loc="upper right")
    parts = ["VMEC2000", "vmec_jax"]
    if include_vmecpp:
        parts.append("VMEC++")
    title_str = " vs ".join(parts)
    title = {
        "all": f"Bundled Example Runtime: {title_str}",
        "fixed": f"Bundled Fixed-Boundary Runtime: {title_str}",
        "freeb": f"Bundled Free-Boundary Runtime: {title_str}",
    }[figure_kind]
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _write_figure(rows: list[dict[str, Any]], outpath: Path, *, figure_kind: str) -> None:
    if figure_kind == "fixed":
        rows = [row for row in rows if not bool(row["lfreeb"])]
    elif figure_kind == "freeb":
        rows = [row for row in rows if bool(row["lfreeb"])]
    if not rows:
        raise ValueError(f"No rows available for figure_kind={figure_kind!r}.")
    sortable = []
    for row in rows:
        runtime_speedup = _speedup(row["cpu_runtime_s"], row["vmec_runtime_s"]) or 0.0
        sortable.append((runtime_speedup, row["id"], row))
    ordered_rows = [row for _, _, row in sorted(sortable, key=lambda item: (-item[0], item[1]))]

    fig, axes = plt.subplots(1, 1, figsize=(8.0, max(8.0, 0.34 * len(ordered_rows) + 1.2)))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    _draw_speedup_panel(
        axes[0],
        rows=ordered_rows,
        value_key="cpu_runtime_s",
        base_key="vmec_runtime_s",
        title="vmec_jax Speedup vs VMEC2000",
        color="#1f77b4",
    )
    title = {
        "all": "Bundled Example Speedup: vmec_jax vs VMEC2000",
        "fixed": "Bundled Fixed-Boundary Speedup: optimized vmec_jax CLI vs VMEC2000",
        "freeb": "Bundled Free-Boundary Speedup: vmec_jax vs VMEC2000",
    }[figure_kind]
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cpu-summary",
        type=Path,
        nargs="+",
        required=True,
        help="Summary JSON from the local CPU run (must include VMEC2000 + vmec_jax CPU).",
    )
    p.add_argument(
        "--gpu-summary",
        type=Path,
        nargs="*",
        default=[],
        help="One or more summary JSON files from GPU vmec_jax runs. Later files override earlier case rows.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "docs" / "_static" / "figures",
    )
    p.add_argument(
        "--figure-out",
        type=Path,
        default=None,
        help="Optional explicit output path for the figure (defaults to outdir/readme_runtime_compare.png).",
    )
    p.add_argument(
        "--table-out",
        type=Path,
        default=REPO_ROOT / "outputs" / "readme_runtime_table.md",
    )
    p.add_argument(
        "--figure-kind",
        choices=("all", "fixed", "freeb"),
        default="fixed",
        help="Subset used in the README figure. The markdown table always keeps all rows.",
    )
    p.add_argument(
        "--plot-mode",
        choices=("runtime", "speedup"),
        default="runtime",
        help="Figure style for the README plot.",
    )
    args = p.parse_args()

    cpu_summaries = [_load_summary(path.expanduser().resolve()) for path in args.cpu_summary]
    gpu_summaries = [_load_summary(path.expanduser().resolve()) for path in args.gpu_summary]
    rows = _collect_records(cpu_summaries, gpu_summaries)

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    table_out = args.table_out.expanduser().resolve()
    table_out.parent.mkdir(parents=True, exist_ok=True)

    fig_out = (args.figure_out.expanduser().resolve() if args.figure_out is not None else (outdir / "readme_runtime_compare.png"))
    if str(args.plot_mode) == "speedup":
        _write_figure(rows, fig_out, figure_kind=str(args.figure_kind))
    else:
        _write_runtime_figure(rows, fig_out, figure_kind=str(args.figure_kind))
    _write_markdown_table(rows, table_out)
    print(f"figure={fig_out}")
    print(f"table={table_out}")


if __name__ == "__main__":
    main()

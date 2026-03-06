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
    gpu_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cpu_cases: dict[str, dict[str, Any]] = {}
    cpu_results: dict[str, dict[str, Any]] = {}
    gpu_results: dict[str, dict[str, Any]] = {}

    for summary in cpu_summaries:
        for rec in summary.get("cases", []):
            cpu_cases[str(rec["id"])] = dict(rec)
        for rec in summary.get("results", []):
            case_id = str(rec.get("case_id"))
            if rec.get("backend") == "vmec2000":
                cpu_results.setdefault(case_id, {})["vmec2000"] = rec
            elif rec.get("backend") == "vmec_jax":
                cpu_results.setdefault(case_id, {})["cpu"] = rec

    for summary in gpu_summaries:
        for rec in summary.get("results", []):
            case_id = str(rec.get("case_id"))
            if rec.get("backend") == "vmec_jax":
                gpu_results.setdefault(case_id, {})["gpu"] = rec

    rows: list[dict[str, Any]] = []
    for case_id in sorted(cpu_cases):
        case = cpu_cases[case_id]
        vmec = cpu_results.get(case_id, {}).get("vmec2000")
        cpu = cpu_results.get(case_id, {}).get("cpu")
        gpu = gpu_results.get(case_id, {}).get("gpu")
        vmec_rt = None if vmec is None else float(vmec.get("runtime_s", vmec.get("time_real_s", np.nan)))
        cpu_rt = None if cpu is None else float(cpu.get("runtime_s", cpu.get("time_real_s", np.nan)))
        gpu_rt = None if gpu is None else float(gpu.get("runtime_s", gpu.get("time_real_s", np.nan)))
        vmec_mem = _mem_bytes(vmec)
        cpu_mem = _mem_bytes(cpu)
        gpu_mem = _mem_bytes(gpu)
        rows.append(
            {
                "id": case_id,
                "lfreeb": bool(case.get("lfreeb", False)),
                "lasym": bool(case.get("lasym", False)),
                "axisymmetric": bool(case.get("axisymmetric", False)),
                "vmec2000": vmec,
                "cpu": cpu,
                "gpu": gpu,
                "vmec_runtime_s": vmec_rt,
                "cpu_runtime_s": cpu_rt,
                "gpu_runtime_s": gpu_rt,
                "vmec_mem_bytes": vmec_mem,
                "cpu_mem_bytes": cpu_mem,
                "gpu_mem_bytes": gpu_mem,
            }
        )
    return rows


def _ratio(value: float | int | None, baseline: float | int | None) -> float | None:
    if value is None or baseline is None:
        return None
    if float(baseline) <= 0.0:
        return None
    return float(value) / float(baseline)


def _write_markdown_table(rows: list[dict[str, Any]], outpath: Path) -> None:
    header = [
        "| Example | Boundary | Topology | LASYM | VMEC2000 runtime | VMEC2000 memory | vmec_jax CPU runtime | vmec_jax CPU memory | vmec_jax GPU runtime | vmec_jax GPU memory |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines = list(header)
    for row in rows:
        boundary = "free" if row["lfreeb"] else "fixed"
        topology = "axisym" if row["axisymmetric"] else "non-axisym"
        lasym = "true" if row["lasym"] else "false"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    boundary,
                    topology,
                    lasym,
                    _format_seconds(row["vmec_runtime_s"]),
                    _format_gib(row["vmec_mem_bytes"]),
                    _format_seconds(row["cpu_runtime_s"]),
                    _format_gib(row["cpu_mem_bytes"]),
                    _format_seconds(row["gpu_runtime_s"]),
                    _format_gib(row["gpu_mem_bytes"]),
                ]
            )
            + " |"
        )
    outpath.write_text("\n".join(lines) + "\n")


def _draw_ratio_panel(ax, *, rows: list[dict[str, Any]], value_key: str, base_key: str, title: str, xlabel: str) -> None:
    labels = [row["id"] for row in rows]
    cpu_ratio = np.array([_ratio(row[value_key.format(runner="cpu")], row[base_key]) or np.nan for row in rows], dtype=float)
    gpu_ratio = np.array([_ratio(row[value_key.format(runner="gpu")], row[base_key]) or np.nan for row in rows], dtype=float)
    y = np.arange(len(rows), dtype=float)

    ax.barh(y + 0.18, cpu_ratio, height=0.34, color="#1f77b4", label="vmec_jax CPU")
    ax.barh(y - 0.18, gpu_ratio, height=0.34, color="#ff7f0e", label="vmec_jax GPU")
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25, which="both")


def _write_figure(rows: list[dict[str, Any]], outpath: Path) -> None:
    sortable = []
    for row in rows:
        runtime_ratio = max(
            _ratio(row["cpu_runtime_s"], row["vmec_runtime_s"]) or 0.0,
            _ratio(row["gpu_runtime_s"], row["vmec_runtime_s"]) or 0.0,
        )
        sortable.append((runtime_ratio, row["id"], row))
    ordered_rows = [row for _, _, row in sorted(sortable, key=lambda item: (item[0], item[1]))]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, max(8.4, 0.36 * len(ordered_rows) + 1.4)), sharey=True)
    _draw_ratio_panel(
        axes[0],
        rows=ordered_rows,
        value_key="{runner}_runtime_s",
        base_key="vmec_runtime_s",
        title="Runtime Ratio vs VMEC2000",
        xlabel="ratio (log scale)",
    )
    _draw_ratio_panel(
        axes[1],
        rows=ordered_rows,
        value_key="{runner}_mem_bytes",
        base_key="vmec_mem_bytes",
        title="Memory Ratio vs VMEC2000",
        xlabel="ratio (log scale)",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2, frameon=False)
    fig.suptitle("Bundled Example Benchmarks: vmec_jax vs VMEC2000", y=0.985)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.95))
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
        nargs="+",
        required=True,
        help="One or more summary JSON files from GPU vmec_jax runs. Later files override earlier case rows.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "docs" / "_static" / "figures",
    )
    p.add_argument(
        "--table-out",
        type=Path,
        default=REPO_ROOT / "outputs" / "readme_runtime_table.md",
    )
    args = p.parse_args()

    cpu_summaries = [_load_summary(path.expanduser().resolve()) for path in args.cpu_summary]
    gpu_summaries = [_load_summary(path.expanduser().resolve()) for path in args.gpu_summary]
    rows = _collect_records(cpu_summaries, gpu_summaries)

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    table_out = args.table_out.expanduser().resolve()
    table_out.parent.mkdir(parents=True, exist_ok=True)

    fig_out = outdir / "readme_runtime_compare.png"
    _write_figure(rows, fig_out)
    _write_markdown_table(rows, table_out)
    print(f"figure={fig_out}")
    print(f"table={table_out}")


if __name__ == "__main__":
    main()

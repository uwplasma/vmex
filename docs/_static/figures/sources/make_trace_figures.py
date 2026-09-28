#!/usr/bin/env python3
"""Regenerate the two README ``--trace`` figures.

- ``readme_trace_benchmark.webp`` plots ``benchmarks/trace_cross_code.json``:
  loss fraction against time for VMEX, SIMPLE and SIMSOPT, and their runtimes
  without compilation.  The record comes from ``benchmarks/trace_cross_code.py``.
- ``readme_trace_output.webp`` is the six-panel ``*_trace.png`` that
  ``vmex --trace`` writes, for the VMEX run of that record.  The trace is
  deterministic, so it is re-run from the recorded settings, and the lost count
  is checked against the record.  The title shows the recorded wall time.

Usage::

    python docs/_static/figures/sources/make_trace_figures.py WOUT

``WOUT`` is the record's equilibrium (``wout_n3are_R7.75B5.7.nc``, from the
ESSOS or SIMSOPT test files).
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIGURES = REPO / "docs" / "_static" / "figures"
RECORD = REPO / "benchmarks" / "trace_cross_code.json"
COLORS = {"VMEX": "#2a78d6", "SIMPLE": "#1baf7a", "SIMSOPT": "#e0823d"}
STYLES = {"VMEX": "-", "SIMPLE": "--", "SIMSOPT": ":"}


def _webp(fig_or_png, out: Path, width: int, quality: int = 80) -> None:
    from PIL import Image

    image = Image.open(fig_or_png).convert("RGB")
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image.save(out, "WEBP", quality=quality, method=6)
    print(out.name, out.stat().st_size, "bytes")


def benchmark_figure(record: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = 1e3 * np.asarray(record["times"])
    fig, (a_f, a_t) = plt.subplots(
        1, 2, figsize=(7.2, 2.5), dpi=150, gridspec_kw={"width_ratios": [1.7, 1]}, layout="constrained"
    )
    names = list(record["codes"])
    for name in names:
        c = record["codes"][name]
        f = 100 * np.asarray(c["curve"])
        sig = 100 * np.sqrt(f / 100 * (1 - f / 100) / record["particles"])
        a_f.fill_between(t, f - sig, f + sig, color=COLORS[name], alpha=0.15, lw=0)
        a_f.plot(
            t,
            f,
            STYLES[name],
            color=COLORS[name],
            lw=1.6,
            label=f"{name} {100 * c['loss_fraction']:.1f} ± {100 * c['sigma']:.1f} %",
        )
    a_f.set(
        xlabel="time [ms]",
        ylabel="loss fraction [%]",
        xlim=(0, t[-1]),
        ylim=(0, None),
        title=f"{record['particles']} alphas, ARIES-CS, s = {record['s']:.3f}",
    )
    a_f.legend(frameon=False, loc="lower right", fontsize=8)
    runs = [record["codes"][n]["run_s"] for n in names]
    bars = a_t.barh(names, runs, color=[COLORS[n] for n in names])
    a_t.bar_label(bars, labels=[f"{r:.0f} s" for r in runs], padding=3, fontsize=8)
    a_t.set(
        xscale="log",
        xlim=(min(runs) / 2, max(runs) * 8),
        xlabel="runtime without compile [s]",
        title=f"{record['cores']} CPU cores each",
    )
    a_t.invert_yaxis()
    for ax in (a_f, a_t):
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    _webp(buf, FIGURES / "readme_trace_benchmark.webp", width=1080)


def output_figure(record: dict, wout: str) -> None:
    os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={record['cores']}")
    from vmex.core.plotting import plot_tracing
    from vmex.core.tracing import trace_alphas

    vmex = record["codes"]["VMEX"]
    result = trace_alphas(
        wout,
        tmax=record["tmax"],
        nparticles=record["particles"],
        s=record["s"],
        scale=None,
        times_to_trace=len(record["times"]),
    )
    if result.particles_lost != vmex["lost"]:
        raise SystemExit(f"re-run lost {result.particles_lost}, the record {vmex['lost']}")
    result = dataclasses.replace(result, wall_time_s=vmex["wall_s"])
    with tempfile.TemporaryDirectory() as tmp:
        png = plot_tracing(result, tmp, name="n3are")["summary"]
        _webp(png, FIGURES / "readme_trace_output.webp", width=1200, quality=70)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wout", help="wout_n3are_R7.75B5.7.nc")
    args = ap.parse_args()
    record = json.loads(RECORD.read_text())
    if Path(args.wout).name != record["wout"]:
        raise SystemExit(f"the record traced {record['wout']}")
    benchmark_figure(record)
    output_figure(record, args.wout)


if __name__ == "__main__":
    main()

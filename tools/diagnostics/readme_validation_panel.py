"""Generate the compact README validation and test-status panel.

The defaults are intentionally explicit snapshots from the latest required CI
run used to update the README.  Override them from the command line when a new
run becomes the reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_JOBS = {
    "parity dry-run": 0.08,
    "docs full": 0.57,
    "build+docs": 0.65,
    "physics smoke": 3.90,
    "fast py3.10": 6.57,
    "fast py3.12": 8.77,
    "fast py3.11": 10.33,
}


def _parse_job(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("job entries must use NAME=MINUTES")
    name, minutes = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("job name cannot be empty")
    try:
        runtime = float(minutes)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid runtime in {value!r}") from exc
    return name, runtime


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.22)


def build_panel(
    *,
    outpath: Path,
    coverage: float,
    coverage_gate: float,
    coverage_target: float,
    local_fast_passed: int,
    local_fast_skipped: int,
    local_fast_minutes: float,
    job_minutes: dict[str, float],
    ci_target_minutes: float,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )

    fig = plt.figure(figsize=(12.0, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.45], height_ratios=[1.0, 1.0])
    ax_cov = fig.add_subplot(gs[0, 0])
    ax_tests = fig.add_subplot(gs[1, 0])
    ax_jobs = fig.add_subplot(gs[:, 1])

    # Coverage gate.
    ax_cov.barh([0], [coverage_target], color="#e9ecef", height=0.5, label="95% target")
    ax_cov.barh([0], [coverage], color="#2b8a3e", height=0.5, label="current")
    ax_cov.axvline(coverage_gate, color="#1c3f95", lw=2.0, ls="--", label=f"CI gate {coverage_gate:.0f}%")
    ax_cov.set_xlim(0.0, 100.0)
    ax_cov.set_yticks([])
    ax_cov.set_xlabel("line coverage (%)")
    ax_cov.set_title("Required coverage gate")
    ax_cov.text(
        min(coverage + 1.0, 96.0),
        0,
        f"{coverage:.2f}%",
        va="center",
        ha="left",
        fontweight="bold",
        color="#1b4332",
    )
    ax_cov.legend(loc="lower right", frameon=False, fontsize=8)
    _style_axes(ax_cov)

    # Local required test result.
    colors = ["#1864ab", "#adb5bd"]
    ax_tests.bar([0, 1], [local_fast_passed, local_fast_skipped], color=colors, width=0.58)
    ax_tests.set_xticks([0, 1], ["passed", "skipped"])
    ax_tests.set_ylabel("tests")
    ax_tests.set_title(f"Local required fast suite ({local_fast_minutes:.2f} min)")
    for idx, value in enumerate([local_fast_passed, local_fast_skipped]):
        ax_tests.text(idx, value + max(local_fast_passed, 1) * 0.025, str(value), ha="center", va="bottom", fontweight="bold")
    _style_axes(ax_tests)

    # Required CI runtime matrix.
    names = list(job_minutes)
    values = np.asarray([job_minutes[name] for name in names], dtype=float)
    y = np.arange(len(names), dtype=float)
    bar_colors = ["#2b8a3e" if val <= ci_target_minutes else "#c92a2a" for val in values]
    ax_jobs.barh(y, values, color=bar_colors, height=0.68)
    ax_jobs.axvline(ci_target_minutes, color="black", lw=1.3, ls="--", alpha=0.8)
    ax_jobs.text(
        ci_target_minutes,
        -0.72,
        f"{ci_target_minutes:.0f} min target",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
    )
    ax_jobs.set_yticks(y, names)
    ax_jobs.invert_yaxis()
    ax_jobs.set_xlabel("wall time (minutes)")
    ax_jobs.set_title("Latest required GitHub Actions gates")
    for yi, val in zip(y, values, strict=True):
        ax_jobs.text(val + 0.12, yi, f"{val:.2f}", va="center", ha="left", fontsize=9)
    ax_jobs.set_xlim(0.0, max(float(np.nanmax(values)) + 1.3, ci_target_minutes + 1.0))
    _style_axes(ax_jobs)

    fig.suptitle("vmec-jax validation status: fast, physics-based gates", fontsize=15, fontweight="bold")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/figures/readme_validation_status.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--coverage", type=float, default=58.82)
    parser.add_argument("--coverage-gate", type=float, default=58.0)
    parser.add_argument("--coverage-target", type=float, default=95.0)
    parser.add_argument("--local-fast-passed", type=int, default=316)
    parser.add_argument("--local-fast-skipped", type=int, default=21)
    parser.add_argument("--local-fast-minutes", type=float, default=2.33)
    parser.add_argument("--ci-target-minutes", type=float, default=10.0)
    parser.add_argument(
        "--job",
        action="append",
        type=_parse_job,
        default=None,
        metavar="NAME=MINUTES",
        help="Override CI job runtime. Can be passed multiple times.",
    )
    args = parser.parse_args(argv)

    jobs = dict(DEFAULT_JOBS)
    if args.job:
        jobs = dict(args.job)

    build_panel(
        outpath=args.output,
        coverage=float(args.coverage),
        coverage_gate=float(args.coverage_gate),
        coverage_target=float(args.coverage_target),
        local_fast_passed=int(args.local_fast_passed),
        local_fast_skipped=int(args.local_fast_skipped),
        local_fast_minutes=float(args.local_fast_minutes),
        job_minutes=jobs,
        ci_target_minutes=float(args.ci_target_minutes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

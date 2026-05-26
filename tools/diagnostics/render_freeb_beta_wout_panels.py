#!/usr/bin/env python
"""Render publication-style finite-beta WOUT comparison panels.

The renderer intentionally consumes WOUT files, not cached arrays, so the
figures are reproducible from the same artifacts used by VMEC/VMEC2000 parity
checks.  It is suitable for free-boundary beta scans where the key evidence is
the evolution of the iota profile, Shafranov/geometric shifts in cross-section,
and LCFS |B| contours.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vmec_jax.plotting import (  # noqa: E402
    bmag_from_wout_physical,
    prepare_matplotlib_3d,
    surface_rz_from_wout_physical,
)
from vmec_jax.wout import read_wout  # noqa: E402


def _import_matplotlib():
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _fsq_total(wout: Any) -> float:
    return float(getattr(wout, "fsqr", np.nan)) + float(getattr(wout, "fsqz", np.nan)) + float(
        getattr(wout, "fsql", np.nan)
    )


def _beta_percent(wout: Any) -> float:
    for name in ("betatotal", "beta_total"):
        value = getattr(wout, name, None)
        if value is not None:
            return 100.0 * float(value)
    return float("nan")


def _mean_iota(wout: Any) -> float:
    values = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", [])), dtype=float)
    values = values[np.isfinite(values)]
    if values.size > 1:
        values = values[1:]
    return float(np.mean(values)) if values.size else float("nan")


def _parse_wout_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path).expanduser().resolve()
    path = Path(spec).expanduser().resolve()
    return None, path


def _runs_from_summary(path: Path, *, backend: str | None, max_actual_beta: float | None) -> list[tuple[str | None, Path]]:
    payload = json.loads(path.read_text())
    runs = payload.get("runs", [])
    selected: list[tuple[str | None, Path]] = []
    for run in runs:
        if backend is not None and str(run.get("backend", "")).lower() != backend.lower():
            continue
        beta = run.get("beta_proxy_percent", None)
        if beta is not None and max_actual_beta is not None and float(beta) > float(max_actual_beta):
            continue
        wout = run.get("wout")
        if not wout:
            continue
        wout_path = Path(str(wout)).expanduser().resolve()
        if not wout_path.exists():
            # Some JSON was written from /private/tmp while the visible path is
            # /tmp.  Keep this normalization local to plotting.
            alt = Path(str(wout).replace("/private/tmp/", "/tmp/")).expanduser().resolve()
            if alt.exists():
                wout_path = alt
        if not wout_path.exists():
            continue
        nominal = run.get("nominal_beta_percent", None)
        label = f"nominal {float(nominal):g}%" if nominal is not None else None
        selected.append((label, wout_path))
    return selected


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[tuple[str | None, Path]] = []
    for summary in args.summary:
        specs.extend(
            _runs_from_summary(
                Path(summary).expanduser().resolve(),
                backend=args.backend,
                max_actual_beta=args.max_actual_beta,
            )
        )
    specs.extend(_parse_wout_spec(item) for item in args.wout)
    if not specs:
        raise SystemExit("No WOUT files were selected. Use --summary or --wout.")

    cases: list[dict[str, Any]] = []
    for user_label, path in specs:
        wout = read_wout(path)
        beta = _beta_percent(wout)
        label = user_label or f"{beta:.2f}%"
        if user_label and np.isfinite(beta):
            label = f"{user_label}\nactual {beta:.2f}%"
        cases.append(
            {
                "label": label,
                "path": path,
                "wout": wout,
                "beta_percent": beta,
                "aspect": float(getattr(wout, "aspect", np.nan)),
                "mean_iota": _mean_iota(wout),
                "fsq_total": _fsq_total(wout),
            }
        )
    cases.sort(key=lambda item: (np.nan_to_num(item["beta_percent"], nan=1.0e99), str(item["path"])))
    return cases


def _write_summary_csv(cases: list[dict[str, Any]], outdir: Path, stem: str) -> Path:
    out = outdir / f"{stem}_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "path", "actual_beta_percent", "aspect", "mean_iota", "fsq_total", "ns", "mpol", "ntor", "nfp"],
        )
        writer.writeheader()
        for case in cases:
            wout = case["wout"]
            writer.writerow(
                {
                    "label": case["label"].replace("\n", " "),
                    "path": str(case["path"]),
                    "actual_beta_percent": f"{case['beta_percent']:.8g}",
                    "aspect": f"{case['aspect']:.8g}",
                    "mean_iota": f"{case['mean_iota']:.8g}",
                    "fsq_total": f"{case['fsq_total']:.8g}",
                    "ns": int(getattr(wout, "ns", -1)),
                    "mpol": int(getattr(wout, "mpol", -1)),
                    "ntor": int(getattr(wout, "ntor", -1)),
                    "nfp": int(getattr(wout, "nfp", -1)),
                }
            )
    return out


def _set_equal_axis(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")


def render_panel(cases: list[dict[str, Any]], *, title: str, outdir: Path, stem: str) -> list[Path]:
    plt = _import_matplotlib()
    ncols = len(cases)
    if ncols > 6:
        raise SystemExit("Too many WOUTs for one publication panel; pass at most 6 cases.")

    fig = plt.figure(figsize=(max(12.0, 3.35 * ncols), 10.8), constrained_layout=True)
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.35, 1.35])

    ax_iota = fig.add_subplot(gs[0, :])
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.1, 0.9, ncols)]
    for color, case in zip(colors, cases, strict=True):
        wout = case["wout"]
        s = np.linspace(0.0, 1.0, int(getattr(wout, "ns", len(np.asarray(wout.iotaf)))))
        iota = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas")), dtype=float)
        ax_iota.plot(s, iota, color=color, lw=2.0, label=f"{case['beta_percent']:.2f}%")
    ax_iota.set_xlabel("Normalized toroidal flux s")
    ax_iota.set_ylabel("iota")
    ax_iota.set_title("Rotational-transform profile")
    ax_iota.grid(True, alpha=0.25)
    ax_iota.legend(title="Actual WOUT beta", ncols=min(ncols, 4), fontsize=9)

    theta_cross = np.linspace(0.0, 2.0 * np.pi, 360)
    radial_color = plt.get_cmap("magma")
    for col, case in enumerate(cases):
        wout = case["wout"]
        ax = fig.add_subplot(gs[1, col])
        indices = np.unique(np.rint(np.linspace(1, int(wout.ns) - 1, 8)).astype(int))
        for j, s_idx in enumerate(indices):
            R, Z = surface_rz_from_wout_physical(
                wout,
                theta=theta_cross,
                phi=np.asarray([0.0]),
                s_index=int(s_idx),
            )
            ax.plot(R[:, 0], Z[:, 0], color=radial_color(j / max(1, len(indices) - 1)), lw=1.0)
        _set_equal_axis(ax)
        ax.set_title(f"{case['label']}\n{case['aspect']:.2f} aspect, fsq {case['fsq_total']:.1e}", fontsize=10)

    theta_b = np.linspace(0.0, 2.0 * np.pi, 144)
    for col, case in enumerate(cases):
        wout = case["wout"]
        nfp = max(1, int(getattr(wout, "nfp", 1)))
        phi_b = np.linspace(0.0, 2.0 * np.pi / float(nfp), 144)
        B = bmag_from_wout_physical(
            wout,
            theta=theta_b,
            phi=phi_b,
            s_index=int(wout.ns) - 1,
        )
        ax = fig.add_subplot(gs[2, col])
        levels = np.linspace(float(np.nanmin(B)), float(np.nanmax(B)), 18)
        if not np.all(np.isfinite(levels)) or levels[-1] <= levels[0]:
            levels = np.linspace(float(np.nanmean(B)) - 1.0e-12, float(np.nanmean(B)) + 1.0e-12, 18)
        contours = ax.contour(phi_b, theta_b, B, levels=levels, cmap="viridis", linewidths=0.85)
        fig.colorbar(contours, ax=ax, fraction=0.046, pad=0.035, label="|B|")
        ax.set_xlabel("Physical toroidal angle phi")
        ax.set_ylabel("Poloidal angle theta")
        ax.set_title(f"LCFS |B| contours\nrange {np.nanmin(B):.3g}-{np.nanmax(B):.3g}", fontsize=10)

    fig.suptitle(title, fontsize=15)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        ("png", {"dpi": 220}),
        ("svg", {}),
        ("pdf", {}),
    ):
        out = outdir / f"{stem}.{suffix}"
        fig.savefig(out, **kwargs)
        outputs.append(out)
    plt.close(fig)
    outputs.append(_write_summary_csv(cases, outdir, stem))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", default=[], help="Beta-scan summary.json to read WOUT paths from.")
    parser.add_argument("--wout", action="append", default=[], help="WOUT path, optionally label=/path/to/wout.nc.")
    parser.add_argument("--backend", default=None, help="Backend filter for --summary runs, e.g. direct or mgrid.")
    parser.add_argument("--max-actual-beta", type=float, default=None, help="Drop summary rows above this actual beta percent.")
    parser.add_argument("--title", default="Free-boundary finite-beta scan", help="Figure title.")
    parser.add_argument("--stem", default="freeb_beta_wout_panel", help="Output filename stem.")
    parser.add_argument("--outdir", default="/tmp/freeb_publication_panels", help="Output directory.")
    args = parser.parse_args(argv)

    cases = _load_cases(args)
    outputs = render_panel(cases, title=args.title, outdir=Path(args.outdir).expanduser().resolve(), stem=args.stem)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate the README diagnostic summary figures.

Writes the ``vmex --plot`` summary panel for two bundled finite-pressure
decks and nothing else: ``readme_diagnostics_qa.webp`` for the NFP=2 QA case
(``examples/data/input.nfp2_QA_finite_beta``) and
``readme_diagnostics_summary.webp`` for the NFP=4 QI case
(``examples/data/input.nfp4_QI_finite_beta``), both in
``docs/_static/figures``.

The figure this replaces was committed with no generator and no recorded
input, so nobody could reproduce or check it.  Running this script also
writes ``docs/_static/figures/readme_diagnostics.json``, recording the deck,
its sha256, the solved scalars the README quotes, and the commit, host, and
package versions of the run.  ``tests/test_figure_provenance.py`` fails when
the figure changes without that record.

Usage::

    python docs/_static/figures/sources/make_readme_diagnostics_figures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from benchmarks._provenance import file_sha256, git_state  # noqa: E402

FIGURES = REPO / "docs" / "_static" / "figures"
RECORD = FIGURES / "readme_diagnostics.json"
SCHEMA = "vmex.readme-diagnostics-figures/1"

CASES: dict[str, dict[str, object]] = {
    "qa_finite_beta": {
        "deck": "examples/data/input.nfp2_QA_finite_beta",
        "figure": "readme_diagnostics_qa.webp",
        "description": "finite-pressure NFP=2 QA equilibrium",
    },
    "qi_finite_beta": {
        "deck": "examples/data/input.nfp4_QI_finite_beta",
        "figure": "readme_diagnostics_summary.webp",
        "description": "finite-pressure NFP=4 QI equilibrium",
    },
}

# The 3000 px summary is downsampled to twice the README column width, where
# its panel labels stay legible, and stored as lossy webp to keep the file
# small.  Pillow's encoder is deterministic for fixed pixels and settings.
WEBP_WIDTH = 1760
WEBP_QUALITY = 82


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _cpu_brand() -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_webp(png: Path, out: Path) -> tuple[int, int]:
    """Downsample the rendered summary PNG and store it as compact webp."""
    from PIL import Image

    with Image.open(png) as source:
        image = source.convert("RGB")
    height = round(image.height * WEBP_WIDTH / image.width)
    image = image.resize((WEBP_WIDTH, height), Image.Resampling.LANCZOS)
    image.save(out, "WEBP", quality=WEBP_QUALITY, method=6)
    return WEBP_WIDTH, height


def run_case(key: str) -> dict[str, object]:
    import vmex as vj
    from vmex.core.plotting import plot_summary

    spec = CASES[key]
    deck = REPO / str(spec["deck"])
    inp = vj.VmecInput.from_file(deck)
    result = vj.solve_multigrid(inp, verbose=True)
    if not result.converged:
        raise RuntimeError(f"{key}: solve did not converge; refusing to plot it")
    wout = vj.wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=True,
    )
    out = FIGURES / str(spec["figure"])
    with tempfile.TemporaryDirectory() as tmp:
        wout_path = vj.write_wout(Path(tmp) / f"wout_{key}.nc", wout)
        png = plot_summary(wout_path, Path(tmp) / f"{key}_summary.png")
        pixels = _write_webp(png, out)

    return {
        "key": key,
        "description": spec["description"],
        "deck": spec["deck"],
        "deck_sha256": file_sha256(deck),
        "cli_equivalent": f"vmex {spec['deck']} --plot",
        "figure": f"docs/_static/figures/{spec['figure']}",
        "figure_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "figure_pixels": list(pixels),
        "iterations": int(result.iterations),
        "scalars": {
            "betatotal": float(wout.betatotal),
            "aspect": float(wout.aspect),
            "volume_p": float(wout.volume_p),
            "ns": int(wout.ns),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=sorted(CASES))
    args = parser.parse_args()

    # Record the source state before the run rewrites its own outputs, which
    # would otherwise mark every regenerated figure as a dirty measurement.
    provenance = dict(git_state(REPO))
    cases = [run_case(key) for key in (args.only or list(CASES))]
    for case in cases:
        print(
            f"[diagnostics] {case['key']}: betatotal="
            f"{case['scalars']['betatotal']:.4%} -> {case['figure']}",
            flush=True,
        )

    provenance.update(
        {
            "measurement_date": time.strftime("%Y-%m-%d"),
            "host": f"{platform.system()} {platform.release()} ({platform.machine()}), CPU",
            "cpu": _cpu_brand(),
            "python": platform.python_version(),
            "versions": {
                "vmex": _package_version("vmex"),
                "jax": _package_version("jax"),
                "matplotlib": _package_version("matplotlib"),
            },
        }
    )
    RECORD.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "generator": (
                    "docs/_static/figures/sources/make_readme_diagnostics_figures.py"
                ),
                "provenance": provenance,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[diagnostics] wrote {RECORD.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

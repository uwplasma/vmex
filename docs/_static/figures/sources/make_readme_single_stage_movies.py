#!/usr/bin/env python3
"""Regenerate the two README single-stage movies as small animated WebP.

Runs each example below as written (in a temporary directory, so its output
files never touch the tree), then animates its accepted iterates: the plasma
boundary as a wireframe and the coils as curves, on axes fixed across frames.

- ``readme_single_stage_fixed_boundary.webp`` from
  ``examples/optimization/single_stage_optimization.py`` (fixed-boundary
  vacuum): the boundary comes straight from each iterate's variables;
- ``readme_single_stage_free_boundary.webp`` from
  ``examples/optimization/single_stage_free_boundary_optimization_finite_beta.py``:
  the coils are the only variables, so each frame re-solves the free boundary
  in that iterate's coil field.

Animated WebP plays inline in a GitHub README from a committed file; a video
would need an uploaded attachment.  Needs ``vmex[coils]`` (ESSOS).

Usage::

    python docs/_static/figures/sources/make_readme_single_stage_movies.py
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIGURES = REPO / "docs" / "_static" / "figures"
EXAMPLES = REPO / "examples" / "optimization"

FRAME_MS, LAST_FRAME_MS = 300, 1500
DPI = 72          # the movie's 5.2 x 4.5 inch canvas -> 374 x 324 px
WEBP_QUALITY = 50


def _fixed_boundary(g: dict):
    return lambda u: g["objects_from_x"](g["x0"] + g["scales"] * u)


def _free_boundary(g: dict):
    import jax.numpy as jnp
    import vmex as vj

    def objects(u):
        state = vj.solve_free_boundary_implicit_status(
            g["params"], jnp.asarray(u), g["config"])[0]
        return (g["boundary_surface"](state),
                g["coils_from_x"](jnp.asarray(g["x0"] + g["scales"] * u)))

    return objects


CASES = {
    "fixed": ("single_stage_optimization.py",
              "readme_single_stage_fixed_boundary.webp", _fixed_boundary),
    "free": ("single_stage_free_boundary_optimization_finite_beta.py",
             "readme_single_stage_free_boundary.webp", _free_boundary),
}


def _write_webp(gif: Path, out: Path) -> None:
    from PIL import Image, ImageSequence

    with Image.open(gif) as movie:
        frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(movie)]
    durations = [FRAME_MS] * (len(frames) - 1) + [LAST_FRAME_MS]
    frames[0].save(out, "WEBP", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, quality=WEBP_QUALITY, method=6)


def run_case(key: str) -> Path:
    script, figure, factory = CASES[key]
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            g = runpy.run_path(str(EXAMPLES / script), run_name="__main__")
        except SystemExit as exc:  # a missed target still leaves the history
            raise RuntimeError(f"{script} exited with {exc.code}") from exc
        finally:
            os.chdir(cwd)
        gif = Path(tmp) / "movie.gif"
        g["monitor"].movie(gif, factory(g), dpi=DPI)
        out = FIGURES / figure
        _write_webp(gif, out)
    print(f"[movie] {key}: {out.relative_to(REPO)} ({out.stat().st_size} bytes)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", choices=sorted(CASES))
    args = parser.parse_args()
    if args.only:
        for key in args.only:
            run_case(key)
        return 0
    # One process per example: the two scripts' JAX/ESSOS traces collide
    # when run back to back in one interpreter.
    for key in CASES:
        subprocess.run([sys.executable, __file__, "--only", key], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

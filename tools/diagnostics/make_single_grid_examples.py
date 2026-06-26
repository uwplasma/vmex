"""Generate single-grid VMEC inputs from bundled examples.

This copies bundled inputs from `examples/data` into `examples/data/single_grid`
and rewrites iteration controls to a single explicit grid request:

- NS_ARRAY = 151
- NITER_ARRAY = 5000
- FTOL_ARRAY = 1e-14
- NSTEP = 500

This intentionally avoids multigrid staging (no arrays with multiple entries),
and avoids scalar NS/NITER/FTOL to keep the intent unambiguous for VMEC2000 and
vmec_jax.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

_DROP_KEYS = {
    "PRE_NITER_ARRAY",
    "PRE_NITER",
    "MULTIGRID",
}


def _split_comment(line: str) -> tuple[str, str]:
    if "!" not in line:
        return line, ""
    head, tail = line.split("!", 1)
    return head, "!" + tail


def _key_of_assignment(code: str) -> str | None:
    # Match `KEY = ...` at the start of the statement, ignoring leading spaces.
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", code)
    if m is None:
        return None
    return str(m.group(1)).upper()


def _rewrite_input(text: str, *, ns: int, niter: int, ftol: float, nstep: int) -> str:
    out_lines: list[str] = []
    saw_ns = False
    saw_niter = False
    saw_ftol = False
    saw_nstep = False
    inserted = False

    for line in text.splitlines(True):
        code, comment = _split_comment(line)
        stripped = code.lstrip()
        if stripped.startswith("&") or stripped.strip() in ("/", "&END"):
            out_lines.append(line)
            continue
        if stripped.startswith("!") or not stripped.strip():
            out_lines.append(line)
            continue

        key = _key_of_assignment(code)
        if key is not None and key in _DROP_KEYS:
            continue

        # VMEC2000 uses NS_ARRAY even for single-grid runs. Keep inputs portable
        # by emitting NS_ARRAY=<ns> and dropping any scalar NS assignments.
        if key == "NS":
            continue
        if key == "NS_ARRAY":
            saw_ns = True
            out_lines.append(
                f"  NS_ARRAY = {int(ns)}{(' ' if comment and not comment.startswith(' !') else '')}{comment}"
            )
            if not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            continue
        if key == "NITER":
            # Drop scalar staged value in favor of NITER_ARRAY.
            continue
        if key == "FTOL":
            # Drop scalar staged value in favor of FTOL_ARRAY.
            continue
        if key == "NITER_ARRAY":
            saw_niter = True
            out_lines.append(
                f"  NITER_ARRAY = {int(niter)}{(' ' if comment and not comment.startswith(' !') else '')}{comment}"
            )
            if not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            continue
        if key == "FTOL_ARRAY":
            saw_ftol = True
            out_lines.append(
                f"  FTOL_ARRAY = {float(ftol):.1E}{(' ' if comment and not comment.startswith(' !') else '')}{comment}"
            )
            if not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            continue
        if key == "NSTEP":
            saw_nstep = True
            out_lines.append(
                f"  NSTEP = {int(nstep)}{(' ' if comment and not comment.startswith(' !') else '')}{comment}"
            )
            if not out_lines[-1].endswith("\n"):
                out_lines[-1] += "\n"
            continue

        out_lines.append(line)

    # Insert controls right after &INDATA for determinism if any were missing.
    if not (saw_ns and saw_niter and saw_ftol and saw_nstep):
        final_lines: list[str] = []
        for line in out_lines:
            final_lines.append(line)
            if (not inserted) and line.lstrip().upper().startswith("&INDATA"):
                if not saw_niter:
                    final_lines.append(f"  NITER_ARRAY = {int(niter)}\n")
                if not saw_ftol:
                    final_lines.append(f"  FTOL_ARRAY = {float(ftol):.1E}\n")
                if not saw_ns:
                    final_lines.append(f"  NS_ARRAY = {int(ns)}\n")
                if not saw_nstep:
                    final_lines.append(f"  NSTEP = {int(nstep)}\n")
                inserted = True
        out_lines = final_lines

    return "".join(out_lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=REPO_ROOT / "examples" / "data", help="Source examples data dir.")
    p.add_argument(
        "--dst",
        type=Path,
        default=REPO_ROOT / "examples/data/single_grid",
        help="Destination data dir.",
    )
    p.add_argument("--ns", type=int, default=151)
    p.add_argument("--niter", type=int, default=5000)
    p.add_argument("--ftol", type=float, default=1.0e-14)
    p.add_argument("--nstep", type=int, default=500)
    p.add_argument("--force", action="store_true", help="Overwrite destination if it exists.")
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    if dst.exists():
        if not args.force:
            raise SystemExit(f"destination exists (use --force): {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store"))

    # Rewrite only input.*; keep mgrid and reference wouts available in the copy.
    for input_path in sorted(dst.glob("input.*")):
        raw = input_path.read_text()
        rewritten = _rewrite_input(
            raw, ns=int(args.ns), niter=int(args.niter), ftol=float(args.ftol), nstep=int(args.nstep)
        )
        input_path.write_text(rewritten)

    print(f"wrote={dst}")
    print(f"ns={int(args.ns)} niter={int(args.niter)} ftol={float(args.ftol):.3e} nstep={int(args.nstep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

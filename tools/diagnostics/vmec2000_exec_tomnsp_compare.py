"""Compare VMEC2000 tomnsps dumps against vmec_jax internal tomnsps arrays.

This tool runs the VMEC2000 executable with a tomnsps dump enabled and compares
the dumped internal Fourier force blocks against vmec_jax's tomnsps output
for the same iteration.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


@dataclass(frozen=True)
class TomnspsDump:
    ns: int
    mpol1: int
    ntor: int
    frcc: np.ndarray
    frss: np.ndarray
    fzsc: np.ndarray
    fzcs: np.ndarray
    flsc: np.ndarray
    flcs: np.ndarray


def _parse_dump(path: Path) -> TomnspsDump:
    ns = None
    mpol1 = None
    ntor = None
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ns="):
                ns = int(line.split("=", 1)[1])
                continue
            if line.startswith("mpol1="):
                mpol1 = int(line.split("=", 1)[1])
                continue
            if line.startswith("ntor="):
                ntor = int(line.split("=", 1)[1])
                continue
            if line.startswith("columns:"):
                continue
            rows.append(line)
    if ns is None or mpol1 is None or ntor is None:
        raise ValueError(f"Missing header in tomnsps dump: {path}")

    shape = (ns, mpol1 + 1, ntor + 1)
    frcc = np.zeros(shape, dtype=float)
    frss = np.zeros(shape, dtype=float)
    fzsc = np.zeros(shape, dtype=float)
    fzcs = np.zeros(shape, dtype=float)
    flsc = np.zeros(shape, dtype=float)
    flcs = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 9:
            continue
        js = int(toks[0]) - 1
        m = int(toks[1])
        n = int(toks[2])
        frcc[js, m, n] = float(toks[3].replace("D", "E").replace("d", "E"))
        frss[js, m, n] = float(toks[4].replace("D", "E").replace("d", "E"))
        fzsc[js, m, n] = float(toks[5].replace("D", "E").replace("d", "E"))
        fzcs[js, m, n] = float(toks[6].replace("D", "E").replace("d", "E"))
        flsc[js, m, n] = float(toks[7].replace("D", "E").replace("d", "E"))
        flcs[js, m, n] = float(toks[8].replace("D", "E").replace("d", "E"))

    return TomnspsDump(
        ns=int(ns),
        mpol1=int(mpol1),
        ntor=int(ntor),
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
    )


def _max_abs_rel(vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30) -> tuple[float, float]:
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan")
    idx = int(np.argmax(diff))
    max_abs = float(diff.flat[idx])
    denom = max(eps, float(abs(vmec_vals.flat[idx])))
    return max_abs, max_abs / denom


def _patch_indata(text: str, *, updates: dict[str, str]) -> str:
    lines = text.splitlines()
    in_block = False
    end_idx = None
    found = {k.upper(): False for k in updates}

    key_re = {k.upper(): re.compile(rf"^(\s*){re.escape(k)}\s*=", flags=re.IGNORECASE) for k in updates}

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("&INDATA"):
            in_block = True
            continue
        if in_block and stripped.startswith("/"):
            end_idx = i
            break
        if not in_block:
            continue

        for k_up, pat in key_re.items():
            if pat.match(line):
                indent = pat.match(line).group(1)
                lines[i] = f"{indent}{k_up} = {updates[k_up]}"
                found[k_up] = True

    if end_idx is None:
        return text

    insert_lines = []
    for k_up, v in updates.items():
        if not found[k_up.upper()]:
            insert_lines.append(f"  {k_up.upper()} = {v}")
    if insert_lines:
        lines = lines[:end_idx] + insert_lines + lines[end_idx:]
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


@contextlib.contextmanager
def _env(overrides: dict[str, str]):
    old = dict(os.environ)
    os.environ.update({k: v for k, v in overrides.items() if v is not None})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="shaped_tokamak_pressure")
    p.add_argument("--input", type=str, default=None, help="Path to input.* (overrides --case).")
    p.add_argument("--vmec2000", type=str, default=None, help="Path to xvmec2000 executable.")
    p.add_argument("--iter", type=int, default=1, help="Iteration to dump/compare.")
    p.add_argument("--single-ns", type=int, default=None, help="Force a single grid ns (no multigrid).")
    p.add_argument("--use-input-niter", action="store_true", help="Use input NITER_ARRAY staging.")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.input is None:
        input_path = root / "examples" / "data" / f"input.{args.case}"
    else:
        input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Missing input file: {input_path}")

    vmec2000_exe = (
        Path(args.vmec2000).expanduser().resolve()
        if args.vmec2000 is not None
        else (root.parent / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000")
    )
    if not vmec2000_exe.exists():
        raise SystemExit(f"Missing VMEC2000 executable: {vmec2000_exe}")

    _cfg_in, _indata_in = vj.load_input(input_path)
    ftol_default = float(_indata_in.get_float("FTOL", 1e-10))

    with tempfile.TemporaryDirectory(prefix="vmec2000_tomnsp_") as td:
        workdir = Path(td)
        input_local = workdir / input_path.name
        shutil.copy2(input_path, input_local)

        updates = {"NSTEP": "1"}
        if args.single_ns is not None:
            ns = int(args.single_ns)
            updates |= {
                "NS_ARRAY": f"{ns}",
                "NITER_ARRAY": f"{int(args.iter)}",
                "FTOL_ARRAY": f"{ftol_default:.16e}",
                "NITER": f"{int(args.iter)}",
            }

        input_local.write_text(_patch_indata(input_local.read_text(), updates=updates))

        env = {
            "VMEC_DUMP_TOMNSPS": "1",
            "VMEC_DUMP_ITER": str(int(args.iter)),
            "VMEC_DUMP_DIR": str(workdir),
        }
        cmd = [str(vmec2000_exe), input_local.name]
        subprocess.run(cmd, cwd=workdir, env={**os.environ, **env}, check=False, capture_output=True, text=True)

        vmec_dump_path = workdir / f"tomnsps_iter{int(args.iter)}.dat"
        if not vmec_dump_path.exists():
            raise SystemExit(f"VMEC2000 dump not found: {vmec_dump_path}")
        vmec_dump = _parse_dump(vmec_dump_path)

        # Run vmec_jax with tomnsps dump enabled.
        jax_dump_dir = workdir / "jax"
        jax_dump_dir.mkdir(parents=True, exist_ok=True)
        with _env(
            {
                "VMEC_JAX_DUMP_TOMNSPS": "1",
                "VMEC_JAX_DUMP_ITER": str(int(args.iter)),
                "VMEC_JAX_DUMP_DIR": str(jax_dump_dir),
            }
        ):
            vj.run_fixed_boundary(
                input_path,
                solver="vmec2000_iter",
                max_iter=int(args.iter),
                multigrid_use_input_niter=bool(args.use_input_niter),
                ns_override=int(args.single_ns) if args.single_ns is not None else None,
                verbose=False,
            )

        jax_dump_path = jax_dump_dir / f"tomnsps_raw_iter{int(args.iter)}.npz"
        if not jax_dump_path.exists():
            raise SystemExit(f"vmec_jax dump not found: {jax_dump_path}")
        jax_dump = np.load(jax_dump_path)

        def _arr(name: str, fallback: np.ndarray) -> np.ndarray:
            arr = jax_dump.get(name, fallback)
            return np.asarray(arr, dtype=float)

        frcc = _arr("frcc", np.zeros_like(vmec_dump.frcc))
        frss = _arr("frss", np.zeros_like(vmec_dump.frss))
        fzsc = _arr("fzsc", np.zeros_like(vmec_dump.fzsc))
        fzcs = _arr("fzcs", np.zeros_like(vmec_dump.fzcs))
        flsc = _arr("flsc", np.zeros_like(vmec_dump.flsc))
        flcs = _arr("flcs", np.zeros_like(vmec_dump.flcs))

        print("tomnsps dump comparison (vmec2000 vs vmec_jax)")
        for name, v, j in [
            ("frcc", vmec_dump.frcc, frcc),
            ("frss", vmec_dump.frss, frss),
            ("fzsc", vmec_dump.fzsc, fzsc),
            ("fzcs", vmec_dump.fzcs, fzcs),
            ("flsc", vmec_dump.flsc, flsc),
            ("flcs", vmec_dump.flcs, flcs),
        ]:
            max_abs, max_rel = _max_abs_rel(v, j)
            print(f"  {name:4s}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")


if __name__ == "__main__":
    main()

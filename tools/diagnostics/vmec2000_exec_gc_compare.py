"""Compare VMEC2000 gc dumps against vmec_jax internal force coefficients.

This tool runs the VMEC2000 executable with gc dumps enabled and compares
the dumped internal Fourier force blocks against vmec_jax's gc dump
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
class GcDump:
    ns: int
    mpol1: int
    ntor: int
    ntmax: int
    gcr: np.ndarray
    gcz: np.ndarray
    gcl: np.ndarray


def _parse_gc_dump(path: Path, *, ntmax: int) -> GcDump:
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
        raise ValueError(f"Missing header in gc dump: {path}")

    shape = (ns, mpol1 + 1, ntor + 1, ntmax)
    gcr = np.zeros(shape, dtype=float)
    gcz = np.zeros(shape, dtype=float)
    gcl = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 7:
            continue
        js = int(toks[0]) - 1
        m = int(toks[1])
        n = int(toks[2])
        t = int(toks[3]) - 1
        if t < 0 or t >= ntmax:
            continue
        gcr[js, m, n, t] = float(toks[4].replace("D", "E").replace("d", "E"))
        gcz[js, m, n, t] = float(toks[5].replace("D", "E").replace("d", "E"))
        gcl[js, m, n, t] = float(toks[6].replace("D", "E").replace("d", "E"))

    return GcDump(
        ns=int(ns),
        mpol1=int(mpol1),
        ntor=int(ntor),
        ntmax=int(ntmax),
        gcr=gcr,
        gcz=gcz,
        gcl=gcl,
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


def _max_diff_report(
    vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30
) -> tuple[float, float, tuple[int, ...], float, float]:
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan"), (), float("nan"), float("nan")
    mask = np.isfinite(diff)
    if not bool(np.any(mask)):
        return float("nan"), float("nan"), (), float("nan"), float("nan")
    idx_flat = int(np.argmax(np.where(mask, diff, -np.inf)))
    idx = tuple(int(i) for i in np.unravel_index(idx_flat, diff.shape))
    max_abs = float(diff[idx])
    vmec_v = float(vmec_vals[idx])
    jax_v = float(jax_vals[idx])
    denom = max(eps, abs(vmec_v))
    max_rel = float(max_abs / denom)
    return max_abs, max_rel, idx, vmec_v, jax_v


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
            m = pat.match(line)
            if m:
                indent = m.group(1)
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


def _ntmax_from_cfg(*, ntor: int, lasym: bool) -> int:
    lthreed = ntor > 0
    if lasym:
        return 4 if lthreed else 2
    return 2 if lthreed else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="shaped_tokamak_pressure")
    p.add_argument("--input", type=str, default=None, help="Path to input.* (overrides --case).")
    p.add_argument("--vmec2000", type=str, default=None, help="Path to xvmec2000 executable.")
    p.add_argument("--iter", type=int, default=1, help="Iteration to dump/compare.")
    p.add_argument("--stage", type=str, default="both", choices=("raw", "precond", "both"))
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
    lasym = bool(_indata_in.get_bool("LASYM", False))
    ntor = int(_indata_in.get_int("NTOR", 0))
    ntmax = _ntmax_from_cfg(ntor=ntor, lasym=lasym)
    ftol_default = float(_indata_in.get_float("FTOL", 1e-10))

    with tempfile.TemporaryDirectory(prefix="vmec2000_gc_") as td:
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
            "VMEC_DUMP_GC": "1",
            "VMEC_DUMP_GC_ITER": str(int(args.iter)),
            "VMEC_DUMP_GC_STAGE": str(args.stage),
            "VMEC_DUMP_GC_DIR": str(workdir),
        }
        cmd = [str(vmec2000_exe), input_local.name]
        subprocess.run(cmd, cwd=workdir, env={**os.environ, **env}, check=False, capture_output=True, text=True)

        dumps = {}
        stages = ("raw", "precond") if args.stage == "both" else (args.stage,)
        for stage in stages:
            vmec_dump_path = workdir / f"gc_{stage}_iter{int(args.iter)}.dat"
            if not vmec_dump_path.exists():
                raise SystemExit(f"VMEC2000 dump not found: {vmec_dump_path}")
            dumps[stage] = _parse_gc_dump(vmec_dump_path, ntmax=ntmax)

        jax_dump_dir = workdir / "jax"
        jax_dump_dir.mkdir(parents=True, exist_ok=True)
        with _env(
            {
                "VMEC_JAX_DUMP_GC": "1",
                "VMEC_JAX_DUMP_GC_ITER": str(int(args.iter)),
                "VMEC_JAX_DUMP_GC_STAGE": str(args.stage),
                "VMEC_JAX_DUMP_GC_DIR": str(jax_dump_dir),
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

        for stage in stages:
            jax_dump_path = jax_dump_dir / f"gc_{stage}_iter{int(args.iter)}.npz"
            if not jax_dump_path.exists():
                raise SystemExit(f"vmec_jax dump not found: {jax_dump_path}")
            jax_dump = np.load(jax_dump_path)
            gcr = np.asarray(jax_dump["gcr"], dtype=float)
            gcz = np.asarray(jax_dump["gcz"], dtype=float)
            gcl = np.asarray(jax_dump["gcl"], dtype=float)

            vmec = dumps[stage]
            ns = min(vmec.gcr.shape[0], gcr.shape[0])
            mpol = min(vmec.gcr.shape[1], gcr.shape[1])
            nrange = min(vmec.gcr.shape[2], gcr.shape[2])
            ntmax_use = min(vmec.gcr.shape[3], gcr.shape[3])

            v_gcr = vmec.gcr[:ns, :mpol, :nrange, :ntmax_use]
            v_gcz = vmec.gcz[:ns, :mpol, :nrange, :ntmax_use]
            v_gcl = vmec.gcl[:ns, :mpol, :nrange, :ntmax_use]
            j_gcr = gcr[:ns, :mpol, :nrange, :ntmax_use]
            j_gcz = gcz[:ns, :mpol, :nrange, :ntmax_use]
            j_gcl = gcl[:ns, :mpol, :nrange, :ntmax_use]

            print(f"gc dump comparison ({stage})")
            for name, v, j in [
                ("gcr", v_gcr, j_gcr),
                ("gcz", v_gcz, j_gcz),
                ("gcl", v_gcl, j_gcl),
            ]:
                max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
                if len(idx) == 4:
                    js, m, n, t = idx
                    loc = f"(js={js+1}, m={m}, n={n}, t={t+1})"
                else:
                    loc = f"{idx}"
                print(
                    f"  {name:3s}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  at {loc}  "
                    f"vmec={vmec_v:.8e}  jax={jax_v:.8e}"
                )


if __name__ == "__main__":
    main()

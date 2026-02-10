"""Compare VMEC2000 bcovar half-mesh field dumps against vmec_jax.

This tool runs the VMEC2000 executable with bcovar-related dumps enabled and
diffs them against vmec_jax's bcovar quantities dumped from the same `vmec2000_iter`
workflow.

Primary use: isolate mismatches in `bsq`/`tcon` by determining whether they come
from:
- Jacobian (`r12/sqrtg/tau/ru12/zu12`)
- B contravariant components (`bsupu/bsupv`)
- Metric-based covariant components (`bsubu/bsubv`)
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
class BcovarFieldDump:
    ns: int
    ntheta3: int
    nzeta: int
    phipog: np.ndarray  # (ns,ntheta3,nzeta)
    bsupu: np.ndarray
    bsupv: np.ndarray
    bsubu: np.ndarray
    bsubv: np.ndarray
    bsq: np.ndarray
    r12: np.ndarray | None = None
    tau: np.ndarray | None = None


def _parse_bcovar_fields_dump(path: Path) -> BcovarFieldDump:
    ns = None
    ntheta3 = None
    nzeta = None
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ns="):
                ns = int(line.split("=", 1)[1])
                continue
            if line.startswith("ntheta3="):
                ntheta3 = int(line.split("=", 1)[1])
                continue
            if line.startswith("nzeta="):
                nzeta = int(line.split("=", 1)[1])
                continue
            if line.startswith("columns:"):
                continue
            if line.startswith("phipog") or line.startswith("bsubu"):
                continue
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header fields in bcovar dump: {path}")

    shape = (int(ns), int(ntheta3), int(nzeta))

    def _z():
        return np.zeros(shape, dtype=float)

    phipog = _z()
    bsupu = _z()
    bsupv = _z()
    bsubu = _z()
    bsubv = _z()
    bsq = _z()
    r12 = _z()
    tau = _z()

    for line in rows:
        toks = line.split()
        # 3 ints + 6 floats = 9 tokens (optional +2 for r12/tau).
        if len(toks) < 9:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[3:]]
        phipog[js, lt, lz] = vals[0]
        bsupu[js, lt, lz] = vals[1]
        bsupv[js, lt, lz] = vals[2]
        bsubu[js, lt, lz] = vals[3]
        bsubv[js, lt, lz] = vals[4]
        bsq[js, lt, lz] = vals[5]
        if len(vals) >= 8:
            r12[js, lt, lz] = vals[6]
            tau[js, lt, lz] = vals[7]

    return BcovarFieldDump(
        ns=int(ns),
        ntheta3=int(ntheta3),
        nzeta=int(nzeta),
        phipog=phipog,
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        bsq=bsq,
        r12=r12 if np.any(r12) else None,
        tau=tau if np.any(tau) else None,
    )


def _max_diff_report(vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30):
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan"), (), float("nan"), float("nan")
    idx_flat = int(np.nanargmax(diff))
    idx = tuple(int(i) for i in np.unravel_index(idx_flat, diff.shape))
    max_abs = float(diff[idx])
    vmec_v = float(vmec_vals[idx])
    jax_v = float(jax_vals[idx])
    denom = max(eps, abs(vmec_v))
    max_rel = float(max_abs / denom)
    return max_abs, max_rel, idx, vmec_v, jax_v


def _max_diff_report_mask_axis(vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30):
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    if vmec_vals.ndim < 1 or vmec_vals.shape[0] < 2:
        return _max_diff_report(vmec_vals, jax_vals, eps=eps)
    return _max_diff_report(vmec_vals[1:], jax_vals[1:], eps=eps)


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="shaped_tokamak_pressure")
    p.add_argument("--input", type=str, default=None, help="Path to input.* (overrides --case).")
    p.add_argument("--vmec2000", type=str, default=None, help="Path to xvmec2000 executable.")
    p.add_argument("--iter", type=int, default=2, help="Iteration to dump/compare.")
    p.add_argument("--single-ns", type=int, default=17, help="Force a single grid ns (no multigrid).")
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

    with tempfile.TemporaryDirectory(prefix="vmec2000_bcovar_") as td:
        workdir = Path(td)
        input_local = workdir / input_path.name
        shutil.copy2(input_path, input_local)

        it = int(args.iter)
        ns = int(args.single_ns)
        updates = {"NSTEP": "1"}
        if ns is not None:
            updates |= {
                "NS_ARRAY": f"{ns}",
                "NITER_ARRAY": f"{it}",
                "FTOL_ARRAY": f"{ftol_default:.16e}",
                "NITER": f"{it}",
            }
        input_local.write_text(_patch_indata(input_local.read_text(), updates=updates))

        env = {
            "VMEC_DUMP_PRECOND": "1",
            "VMEC_DUMP_BCOVAR": "1",
            "VMEC_DUMP_ITER": str(it),
            "VMEC_DUMP_DIR": str(workdir),
        }
        subprocess.run([str(vmec2000_exe), input_local.name], cwd=workdir, env={**os.environ, **env}, check=False)

        vmec_fields_path = workdir / f"bcovar_fields_iter{it}.dat"
        if not vmec_fields_path.exists():
            raise SystemExit(f"VMEC2000 bcovar fields dump not found: {vmec_fields_path}")
        vmec_fields = _parse_bcovar_fields_dump(vmec_fields_path)

        # Run vmec_jax with bcovar dump enabled.
        jax_dump_dir = workdir / "jax"
        jax_dump_dir.mkdir(parents=True, exist_ok=True)
        with _env(
            {
                "VMEC_JAX_DUMP_BCOVAR": "1",
                "VMEC_JAX_DUMP_ITER": str(it),
                "VMEC_JAX_DUMP_DIR": str(jax_dump_dir),
            }
        ):
            run = vj.run_fixed_boundary(
                input_path,
                solver="vmec2000_iter",
                max_iter=it,
                multigrid_use_input_niter=bool(args.use_input_niter),
                ns_override=ns,
                verbose=False,
            )

        jax_path = jax_dump_dir / f"bcovar_raw_iter{it}.npz"
        if not jax_path.exists():
            raise SystemExit(f"vmec_jax bcovar dump not found: {jax_path}")
        jax = np.load(jax_path)

        def _arr(name: str) -> np.ndarray:
            return np.asarray(jax[name], dtype=float)

        print("bcovar field parity (vmec2000 vs vmec_jax)")
        pairs = [
            ("phipog", vmec_fields.phipog, _arr("phipog_vmec")),
            ("bsupu", vmec_fields.bsupu, _arr("bsupu")),
            ("bsupv", vmec_fields.bsupv, _arr("bsupv")),
            ("bsubu", vmec_fields.bsubu, _arr("bsubu")),
            ("bsubv", vmec_fields.bsubv, _arr("bsubv")),
            ("bsq", vmec_fields.bsq, _arr("bsq")),
        ]
        if vmec_fields.r12 is not None and vmec_fields.tau is not None:
            pairs.extend(
                [
                    ("r12", vmec_fields.r12, _arr("r12")),
                    ("tau", vmec_fields.tau, _arr("tau")),
                ]
            )
        for name, v, j in pairs:
            max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
            js, lt, lz = idx
            loc = f"(js={js+1}, lt={lt+1}, lz={lz+1})"
            print(
                f"  {name:6s}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} at {loc} "
                f"vmec={vmec_v:.8e} jax={jax_v:.8e}"
            )

        print("bcovar field parity (exclude axis js=1)")
        pairs = [
            ("phipog", vmec_fields.phipog, _arr("phipog_vmec")),
            ("bsupu", vmec_fields.bsupu, _arr("bsupu")),
            ("bsupv", vmec_fields.bsupv, _arr("bsupv")),
            ("bsubu", vmec_fields.bsubu, _arr("bsubu")),
            ("bsubv", vmec_fields.bsubv, _arr("bsubv")),
            ("bsq", vmec_fields.bsq, _arr("bsq")),
        ]
        if vmec_fields.r12 is not None and vmec_fields.tau is not None:
            pairs.extend(
                [
                    ("r12", vmec_fields.r12, _arr("r12")),
                    ("tau", vmec_fields.tau, _arr("tau")),
                ]
            )
        for name, v, j in pairs:
            max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report_mask_axis(v, j)
            js, lt, lz = idx
            loc = f"(js={js+2}, lt={lt+1}, lz={lz+1})"
            print(
                f"  {name:6s}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} at {loc} "
                f"vmec={vmec_v:.8e} jax={jax_v:.8e}"
            )

        # Silence "unused variable" lint for run when users run this in notebooks.
        _ = run


if __name__ == "__main__":
    main()

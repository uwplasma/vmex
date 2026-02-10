"""Scan VMEC2000 vs vmec_jax internal force blocks across iterations.

This tool runs the VMEC2000 executable with tomnsps + gc dumps enabled and
compares the dumped arrays against vmec_jax dumps for the same iteration.
It stops at the first mismatch beyond tolerances.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vmec_jax.api as vj
from tools.diagnostics.vmec2000_exec_gc_compare import _ntmax_from_cfg, _parse_gc_dump
from tools.diagnostics.vmec2000_exec_tomnsp_compare import _parse_dump


def _max_diff(vmec_vals: np.ndarray, jax_vals: np.ndarray) -> tuple[float, float, tuple[int, ...], float, float]:
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
    max_rel = float(max_abs / max(1e-30, abs(vmec_v)))
    return max_abs, max_rel, idx, vmec_v, jax_v


def _matches(vmec_val: float, jax_val: float, *, rtol: float, atol: float) -> bool:
    if np.isnan(vmec_val) and np.isnan(jax_val):
        return True
    if np.isinf(vmec_val) and np.isinf(jax_val) and (np.sign(vmec_val) == np.sign(jax_val)):
        return True
    if not (np.isfinite(vmec_val) and np.isfinite(jax_val)):
        return False
    return abs(vmec_val - jax_val) <= max(atol, rtol * abs(vmec_val))


def _compare_block(name: str, vmec_vals: np.ndarray, jax_vals: np.ndarray, *, rtol: float, atol: float) -> bool:
    max_abs, max_rel, idx, v, jv = _max_diff(vmec_vals, jax_vals)
    ok = _matches(v, jv, rtol=rtol, atol=atol)
    status = "OK" if ok else "MISMATCH"
    print(f"  {name:>6s}: {status}  max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={idx}")
    if not ok:
        print(f"    vmec2000={v:.6e}  vmec_jax={jv:.6e}")
    return ok


def _parse_tomnsp_kernels(path: Path):
    ns = None
    ntheta3 = None
    nzeta = None
    rows = []
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
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header in tomnsps kernels dump: {path}")

    shape = (ns, ntheta3, nzeta, 2)
    blmn = np.zeros(shape, dtype=float)
    clmn = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 14:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        mpar = int(toks[3])
        blmn[js, lt, lz, mpar] = float(toks[12].replace("D", "E").replace("d", "E"))
        clmn[js, lt, lz, mpar] = float(toks[13].replace("D", "E").replace("d", "E"))

    return blmn, clmn


def _parse_bsube(path: Path):
    ns = None
    ntheta3 = None
    nzeta = None
    lamscale = None
    rows = []
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
            if line.startswith("lamscale="):
                try:
                    lamscale = float(line.split("=", 1)[1])
                except ValueError:
                    lamscale = None
                continue
            if line.startswith("columns:"):
                continue
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header in bsube dump: {path}")

    shape = (ns, ntheta3, nzeta)
    bsubu_e = np.zeros(shape, dtype=float)
    bsubv_e = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 5:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        bsubu_e[js, lt, lz] = float(toks[3].replace("D", "E").replace("d", "E"))
        bsubv_e[js, lt, lz] = float(toks[4].replace("D", "E").replace("d", "E"))

    return bsubu_e, bsubv_e, lamscale


def _parse_bsubh(path: Path):
    ns = None
    ntheta3 = None
    nzeta = None
    rows = []
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
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header in bsubh dump: {path}")

    shape = (ns, ntheta3, nzeta)
    bsubu_h = np.zeros(shape, dtype=float)
    bsubv_h = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 5:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        bsubu_h[js, lt, lz] = float(toks[3].replace("D", "E").replace("d", "E"))
        bsubv_h[js, lt, lz] = float(toks[4].replace("D", "E").replace("d", "E"))

    return bsubu_h, bsubv_h


def _parse_bsup(path: Path):
    ns = None
    ntheta3 = None
    nzeta = None
    rows = []
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
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header in bsup dump: {path}")

    shape = (ns, ntheta3, nzeta)
    bsupu = np.zeros(shape, dtype=float)
    bsupv = np.zeros(shape, dtype=float)
    for line in rows:
        toks = line.split()
        if len(toks) < 5:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        bsupu[js, lt, lz] = float(toks[3].replace("D", "E").replace("d", "E"))
        bsupv[js, lt, lz] = float(toks[4].replace("D", "E").replace("d", "E"))
    return bsupu, bsupv


def _parse_bsube_terms(path: Path):
    ns = None
    ntheta3 = None
    nzeta = None
    rows = []
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
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header in bsube terms dump: {path}")

    shape = (ns, ntheta3, nzeta)
    lvv_sh = np.zeros(shape, dtype=float)
    lu0 = np.zeros(shape, dtype=float)
    lu1 = np.zeros(shape, dtype=float)
    phipf = np.zeros(shape, dtype=float)
    bsubu_tmp = np.zeros(shape, dtype=float)
    bsubv_pre = np.zeros(shape, dtype=float)

    for line in rows:
        toks = line.split()
        if len(toks) < 7:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        lvv_sh[js, lt, lz] = float(toks[3].replace("D", "E").replace("d", "E"))
        lu0[js, lt, lz] = float(toks[4].replace("D", "E").replace("d", "E"))
        lu1[js, lt, lz] = float(toks[5].replace("D", "E").replace("d", "E"))
        phipf[js, lt, lz] = float(toks[6].replace("D", "E").replace("d", "E"))
        if len(toks) >= 9:
            bsubu_tmp[js, lt, lz] = float(toks[7].replace("D", "E").replace("d", "E"))
            bsubv_pre[js, lt, lz] = float(toks[8].replace("D", "E").replace("d", "E"))

    return lvv_sh, lu0, lu1, phipf, bsubu_tmp, bsubv_pre

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


class _Env:
    def __init__(self, overrides: dict[str, str]):
        self.overrides = overrides
        self._old = None

    def __enter__(self):
        self._old = dict(os.environ)
        os.environ.update({k: v for k, v in self.overrides.items() if v is not None})
        return self

    def __exit__(self, exc_type, exc, tb):
        os.environ.clear()
        os.environ.update(self._old or {})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--input", type=str, default=None, help="Path to input.* (overrides --case).")
    p.add_argument("--vmec2000", type=str, default=None, help="Path to xvmec2000 executable.")
    p.add_argument("--iter-start", type=int, default=1)
    p.add_argument("--iter-stop", type=int, default=5)
    p.add_argument("--single-ns", type=int, default=None, help="Force a single grid ns (no multigrid).")
    p.add_argument("--use-input-niter", action="store_true", help="Use input NITER_ARRAY staging.")
    p.add_argument("--stage", type=str, default="both", choices=("raw", "precond", "both"))
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--timeout", type=float, default=60.0)
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

    cfg, indata = vj.load_input(input_path)
    ntmax = _ntmax_from_cfg(ntor=int(cfg.ntor), lasym=bool(cfg.lasym))
    ftol_default = float(indata.get_float("FTOL", 1e-10))

    it_start = int(args.iter_start)
    it_stop = int(args.iter_stop)
    if it_stop < it_start:
        it_start, it_stop = it_stop, it_start

    for it in range(it_start, it_stop + 1):
        print(f"[scan] iter={it}", flush=True)
        with tempfile.TemporaryDirectory(prefix="vmec2000_internal_") as td:
            workdir = Path(td)
            input_local = workdir / input_path.name
            shutil.copy2(input_path, input_local)

            updates = {"NSTEP": "1"}
            if args.single_ns is not None:
                ns = int(args.single_ns)
                updates |= {
                    "NS_ARRAY": f"{ns}",
                    "NITER_ARRAY": f"{int(it)}",
                    "FTOL_ARRAY": f"{ftol_default:.16e}",
                    "NITER": f"{int(it)}",
                }
            input_local.write_text(_patch_indata(input_local.read_text(), updates=updates))

            env = {
                "VMEC_DUMP_TOMNSPS": "1",
                "VMEC_DUMP_TOMNSPS_KERNELS": "1",
                "VMEC_DUMP_BSUBE": "1",
                "VMEC_DUMP_BSUBE_TERMS": "1",
                "VMEC_DUMP_BSUBH": "1",
                "VMEC_DUMP_BSUP": "1",
                "VMEC_DUMP_ITER": str(int(it)),
                "VMEC_DUMP_GC": "1",
                "VMEC_DUMP_GC_ITER": str(int(it)),
                "VMEC_DUMP_GC_STAGE": str(args.stage),
                "VMEC_DUMP_DIR": str(workdir),
            }
            cmd = [str(vmec2000_exe), input_local.name]
            subprocess.run(
                cmd,
                cwd=workdir,
                env={**os.environ, **env},
                check=False,
                capture_output=True,
                text=True,
                timeout=float(args.timeout),
            )

            vmec_tomnsp_path = workdir / f"tomnsps_iter{int(it)}.dat"
            if not vmec_tomnsp_path.exists():
                raise SystemExit(f"Missing VMEC2000 tomnsps dump: {vmec_tomnsp_path}")
            vmec_tomnsp = _parse_dump(vmec_tomnsp_path)

            vmec_gc: dict[str, any] = {}
            stages = ("raw", "precond") if args.stage == "both" else (args.stage,)
            for stage in stages:
                path = workdir / f"gc_{stage}_iter{int(it)}.dat"
                if not path.exists():
                    raise SystemExit(f"Missing VMEC2000 gc dump: {path}")
                vmec_gc[stage] = _parse_gc_dump(path, ntmax=ntmax)

            # Run vmec_jax and dump arrays.
            jax_dump_dir = workdir / "jax"
            jax_dump_dir.mkdir(parents=True, exist_ok=True)
            env_jax = {
                "VMEC_JAX_DUMP_TOMNSPS": "1",
                "VMEC_JAX_DUMP_FORCE_KERNELS": "1",
                "VMEC_JAX_DUMP_ITER": str(int(it)),
                "VMEC_JAX_DUMP_DIR": str(jax_dump_dir),
                "VMEC_JAX_DUMP_GC": "1",
                "VMEC_JAX_DUMP_GC_ITER": str(int(it)),
                "VMEC_JAX_DUMP_GC_STAGE": str(args.stage),
                "VMEC_JAX_DUMP_GC_DIR": str(jax_dump_dir),
            }
            with _Env(env_jax):
                vj.run_fixed_boundary(
                    input_path,
                    solver="vmec2000_iter",
                    max_iter=int(it),
                    multigrid_use_input_niter=bool(args.use_input_niter),
                    ns_override=int(args.single_ns) if args.single_ns is not None else None,
                    verbose=False,
                )

            jax_tomnsp_path = jax_dump_dir / f"tomnsps_raw_iter{int(it)}.npz"
            if not jax_tomnsp_path.exists():
                raise SystemExit(f"Missing vmec_jax tomnsps dump: {jax_tomnsp_path}")
            jax_tomnsp = np.load(jax_tomnsp_path)

            ok = True
            print(" tomnsps:", flush=True)
            for name in ("frcc", "frss", "fzsc", "fzcs", "flsc", "flcs"):
                ok &= _compare_block(
                    name,
                    getattr(vmec_tomnsp, name),
                    np.asarray(jax_tomnsp.get(name, np.zeros((0,), dtype=float))),
                    rtol=float(args.rtol),
                    atol=float(args.atol),
                )

            vmec_kernels_path = workdir / f"tomnsps_kernels_iter{int(it)}.dat"
            jax_kernels_path = jax_dump_dir / f"force_kernels_raw_iter{int(it)}.npz"
            if vmec_kernels_path.exists() and jax_kernels_path.exists():
                vmec_blmn, vmec_clmn = _parse_tomnsp_kernels(vmec_kernels_path)
                jax_kernels = np.load(jax_kernels_path)
                jax_blmn = np.stack([jax_kernels["blmn_e"], jax_kernels["blmn_o"]], axis=-1)
                jax_clmn = np.stack([jax_kernels["clmn_e"], jax_kernels["clmn_o"]], axis=-1)
                print(" tomnsps kernels:", flush=True)
                ok &= _compare_block("blmn", vmec_blmn, jax_blmn, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("clmn", vmec_clmn, jax_clmn, rtol=float(args.rtol), atol=float(args.atol))

            vmec_bsube_path = workdir / f"bsube_iter{int(it)}.dat"
            if vmec_bsube_path.exists() and jax_kernels_path.exists():
                vmec_bsubu_e, vmec_bsubv_e, vmec_lamscale = _parse_bsube(vmec_bsube_path)
                jax_kernels = np.load(jax_kernels_path)
                jax_bsubu = np.asarray(jax_kernels.get("bsubu_e_scaled", np.zeros((0,), dtype=float)))
                jax_bsubv = np.asarray(jax_kernels.get("bsubv_e_scaled", np.zeros((0,), dtype=float)))
                print(" bsube (scaled):", flush=True)
                ok &= _compare_block("bsubu_e", vmec_bsubu_e, jax_bsubu, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("bsubv_e", vmec_bsubv_e, jax_bsubv, rtol=float(args.rtol), atol=float(args.atol))
                if vmec_lamscale is not None and "lamscale" in jax_kernels:
                    jax_lamscale = float(np.asarray(jax_kernels["lamscale"]))
                    ok &= _compare_block(
                        "lamscale",
                        np.asarray([vmec_lamscale], dtype=float),
                        np.asarray([jax_lamscale], dtype=float),
                        rtol=float(args.rtol),
                        atol=float(args.atol),
                    )

            vmec_bsubh_path = workdir / f"bsubh_iter{int(it)}.dat"
            if vmec_bsubh_path.exists() and jax_kernels_path.exists():
                vmec_bsubu_h, vmec_bsubv_h = _parse_bsubh(vmec_bsubh_path)
                jax_kernels = np.load(jax_kernels_path)
                jax_bsubu = np.asarray(jax_kernels.get("bsubu", np.zeros((0,), dtype=float)))
                jax_bsubv = np.asarray(jax_kernels.get("bsubv", np.zeros((0,), dtype=float)))
                print(" bsubh (half mesh):", flush=True)
                ok &= _compare_block("bsubu_h", vmec_bsubu_h, jax_bsubu, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("bsubv_h", vmec_bsubv_h, jax_bsubv, rtol=float(args.rtol), atol=float(args.atol))

            vmec_bsup_path = workdir / f"bsup_iter{int(it)}.dat"
            if vmec_bsup_path.exists() and jax_kernels_path.exists():
                vmec_bsupu, vmec_bsupv = _parse_bsup(vmec_bsup_path)
                jax_kernels = np.load(jax_kernels_path)
                jax_bsupu = np.asarray(jax_kernels.get("bsupu", np.zeros((0,), dtype=float)))
                jax_bsupv = np.asarray(jax_kernels.get("bsupv", np.zeros((0,), dtype=float)))
                print(" bsup (half mesh):", flush=True)
                ok &= _compare_block("bsupu", vmec_bsupu, jax_bsupu, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("bsupv", vmec_bsupv, jax_bsupv, rtol=float(args.rtol), atol=float(args.atol))

            vmec_terms_path = workdir / f"bsube_terms_iter{int(it)}.dat"
            if vmec_terms_path.exists() and jax_kernels_path.exists():
                vmec_lvv_sh, vmec_lu0, vmec_lu1, vmec_phipf, vmec_bsubu_tmp, vmec_bsubv_pre = _parse_bsube_terms(
                    vmec_terms_path
                )
                jax_kernels = np.load(jax_kernels_path)
                jax_lvv_sh = np.asarray(jax_kernels.get("lvv_sh", np.zeros((0,), dtype=float)))
                if "lu0_force" in jax_kernels and np.asarray(jax_kernels["lu0_force"]).size:
                    jax_lu0 = np.asarray(jax_kernels["lu0_force"], dtype=float)
                    lu0_label = "lu0_force"
                else:
                    jax_lu0 = np.asarray(jax_kernels.get("lu0_full", np.zeros((0,), dtype=float)))
                    lu0_label = "lu0_full"
                jax_lu1 = np.asarray(jax_kernels.get("lu1_full", np.zeros((0,), dtype=float)))
                jax_bsubu_tmp = np.asarray(jax_kernels.get("bsubu_tmp", np.zeros((0,), dtype=float)))
                jax_bsubv_pre = np.asarray(jax_kernels.get("bsubv_preblend", np.zeros((0,), dtype=float)))
                if "phip_internal" in jax_kernels and np.asarray(jax_kernels["phip_internal"]).size:
                    jax_phipf = np.asarray(jax_kernels["phip_internal"], dtype=float)
                    phip_label = "phip_internal"
                else:
                    jax_phipf = np.asarray(jax_kernels.get("phip_full", np.zeros((0,), dtype=float)))
                    phip_label = "phip_full"
                if jax_phipf.ndim == 1 and vmec_phipf.ndim == 3 and jax_phipf.size == vmec_phipf.shape[0]:
                    jax_phipf = jax_phipf[:, None, None] * np.ones_like(vmec_phipf)
                print(" bsube terms:", flush=True)
                ok &= _compare_block("lvv_sh", vmec_lvv_sh, jax_lvv_sh, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block(lu0_label, vmec_lu0, jax_lu0, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("lu1", vmec_lu1, jax_lu1, rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block(phip_label, vmec_phipf, jax_phipf, rtol=float(args.rtol), atol=float(args.atol))
                if vmec_bsubu_tmp.size and jax_bsubu_tmp.size:
                    ok &= _compare_block("bsubu_tmp", vmec_bsubu_tmp, jax_bsubu_tmp, rtol=float(args.rtol), atol=float(args.atol))
                if vmec_bsubv_pre.size and jax_bsubv_pre.size:
                    ok &= _compare_block("bsubv_pre", vmec_bsubv_pre, jax_bsubv_pre, rtol=float(args.rtol), atol=float(args.atol))

            for stage in stages:
                jax_gc_path = jax_dump_dir / f"gc_{stage}_iter{int(it)}.npz"
                if not jax_gc_path.exists():
                    raise SystemExit(f"Missing vmec_jax gc dump: {jax_gc_path}")
                jax_gc = np.load(jax_gc_path)
                print(f" gc({stage}):", flush=True)
                ok &= _compare_block("gcr", vmec_gc[stage].gcr, np.asarray(jax_gc.get("gcr", np.zeros((0,), dtype=float))), rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("gcz", vmec_gc[stage].gcz, np.asarray(jax_gc.get("gcz", np.zeros((0,), dtype=float))), rtol=float(args.rtol), atol=float(args.atol))
                ok &= _compare_block("gcl", vmec_gc[stage].gcl, np.asarray(jax_gc.get("gcl", np.zeros((0,), dtype=float))), rtol=float(args.rtol), atol=float(args.atol))

            if not ok:
                print(f"[scan] mismatch at iter={it}")
                raise SystemExit(2)

    print("[scan] no mismatches within tolerance")


if __name__ == "__main__":
    main()

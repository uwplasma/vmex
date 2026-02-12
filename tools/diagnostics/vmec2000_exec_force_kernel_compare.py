"""Compare VMEC2000 tomnsps-kernel dumps against vmec_jax force kernels.

This tool runs the VMEC2000 executable with a tomnsps *kernel* dump enabled and
compares the dumped real-space kernel blocks against vmec_jax's force-kernel
arrays for the same iteration.
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
class KernelDump:
    ns: int
    ntheta3: int
    nzeta: int
    armn: np.ndarray
    brmn: np.ndarray
    crmn: np.ndarray
    azmn: np.ndarray
    bzmn: np.ndarray
    czmn: np.ndarray
    arcon: np.ndarray
    azcon: np.ndarray
    blmn: np.ndarray
    clmn: np.ndarray


def _parse_bcovar_fields_dump(path: Path):
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

    return {
        "phipog": phipog,
        "bsupu": bsupu,
        "bsupv": bsupv,
        "bsubu": bsubu,
        "bsubv": bsubv,
        "bsq": bsq,
        "r12": r12 if np.any(r12) else None,
        "tau": tau if np.any(tau) else None,
    }


def _parse_kernel_dump(path: Path) -> KernelDump:
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
        raise ValueError(f"Missing header in kernel dump: {path}")

    shape = (ns, ntheta3, nzeta, 2)
    def _zeros():
        return np.zeros(shape, dtype=float)

    armn = _zeros()
    brmn = _zeros()
    crmn = _zeros()
    azmn = _zeros()
    bzmn = _zeros()
    czmn = _zeros()
    arcon = _zeros()
    azcon = _zeros()
    blmn = _zeros()
    clmn = _zeros()

    for line in rows:
        toks = line.split()
        # 4 integer columns + 10 float columns = 14 tokens total.
        if len(toks) < 14:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        mp = int(toks[3])
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[4:14]]
        armn[js, lt, lz, mp] = vals[0]
        brmn[js, lt, lz, mp] = vals[1]
        crmn[js, lt, lz, mp] = vals[2]
        azmn[js, lt, lz, mp] = vals[3]
        bzmn[js, lt, lz, mp] = vals[4]
        czmn[js, lt, lz, mp] = vals[5]
        arcon[js, lt, lz, mp] = vals[6]
        azcon[js, lt, lz, mp] = vals[7]
        blmn[js, lt, lz, mp] = vals[8]
        clmn[js, lt, lz, mp] = vals[9]

    return KernelDump(
        ns=int(ns),
        ntheta3=int(ntheta3),
        nzeta=int(nzeta),
        armn=armn,
        brmn=brmn,
        crmn=crmn,
        azmn=azmn,
        bzmn=bzmn,
        czmn=czmn,
        arcon=arcon,
        azcon=azcon,
        blmn=blmn,
        clmn=clmn,
    )


def _max_diff_report(vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30):
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan"), (), float("nan"), float("nan")
    idx_flat = int(np.argmax(diff))
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

    with tempfile.TemporaryDirectory(prefix="vmec2000_kernel_") as td:
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
            "VMEC_DUMP_TOMNSPS_KERNELS": "1",
            "VMEC_DUMP_TOMNSPS": "1",
            "VMEC_DUMP_BCOVAR": "1",
            "VMEC_DUMP_ITER": str(int(args.iter)),
            "VMEC_DUMP_DIR": str(workdir),
        }
        cmd = [str(vmec2000_exe), input_local.name]
        subprocess.run(cmd, cwd=workdir, env={**os.environ, **env}, check=False, capture_output=True, text=True)

        vmec_dump_path = workdir / f"tomnsps_kernels_iter{int(args.iter)}.dat"
        if not vmec_dump_path.exists():
            candidates = sorted(workdir.glob(f"tomnsps_kernels_ns*_iter{int(args.iter)}.dat"))
            if candidates:
                vmec_dump_path = candidates[-1]
            else:
                raise SystemExit(f"VMEC2000 kernel dump not found: {vmec_dump_path}")
        vmec_dump = _parse_kernel_dump(vmec_dump_path)

        vmec_bcovar_path = workdir / f"bcovar_fields_iter{int(args.iter)}.dat"
        vmec_bcovar = None
        if vmec_bcovar_path.exists():
            vmec_bcovar = _parse_bcovar_fields_dump(vmec_bcovar_path)

        # Run vmec_jax with force-kernel dump enabled.
        jax_dump_dir = workdir / "jax"
        jax_dump_dir.mkdir(parents=True, exist_ok=True)
        with _env(
            {
                "VMEC_JAX_DUMP_FORCE_KERNELS": "1",
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

        jax_dump_path = jax_dump_dir / f"force_kernels_raw_iter{int(args.iter)}.npz"
        if not jax_dump_path.exists():
            candidates = sorted(jax_dump_dir.glob(f"force_kernels_raw_ns*_iter{int(args.iter)}.npz"))
            if candidates:
                jax_dump_path = candidates[-1]
            else:
                raise SystemExit(f"vmec_jax kernel dump not found: {jax_dump_path}")
        jax_dump = np.load(jax_dump_path)

        def _arr(name: str, fallback: np.ndarray) -> np.ndarray:
            arr = jax_dump.get(name, fallback)
            return np.asarray(arr, dtype=float)

        def _compare(name: str, v: np.ndarray, j: np.ndarray) -> None:
            max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
            if len(idx) == 3:
                js, lt, lz = idx
                loc = f"(js={js+1}, lt={lt+1}, lz={lz+1})"
            else:
                loc = f"{idx}"
            print(
                f"  {name:6s}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  at {loc}  "
                f"vmec={vmec_v:.8e}  jax={jax_v:.8e}"
            )

        print("tomnsps kernel comparison (vmec2000 vs vmec_jax)")
        pairs = [
            ("armn_e", vmec_dump.armn[:, :, :, 0], _arr("armn_e", np.zeros_like(vmec_dump.armn[:, :, :, 0]))),
            ("armn_o", vmec_dump.armn[:, :, :, 1], _arr("armn_o", np.zeros_like(vmec_dump.armn[:, :, :, 1]))),
            ("brmn_e", vmec_dump.brmn[:, :, :, 0], _arr("brmn_e", np.zeros_like(vmec_dump.brmn[:, :, :, 0]))),
            ("brmn_o", vmec_dump.brmn[:, :, :, 1], _arr("brmn_o", np.zeros_like(vmec_dump.brmn[:, :, :, 1]))),
            ("crmn_e", vmec_dump.crmn[:, :, :, 0], _arr("crmn_e", np.zeros_like(vmec_dump.crmn[:, :, :, 0]))),
            ("crmn_o", vmec_dump.crmn[:, :, :, 1], _arr("crmn_o", np.zeros_like(vmec_dump.crmn[:, :, :, 1]))),
            ("azmn_e", vmec_dump.azmn[:, :, :, 0], _arr("azmn_e", np.zeros_like(vmec_dump.azmn[:, :, :, 0]))),
            ("azmn_o", vmec_dump.azmn[:, :, :, 1], _arr("azmn_o", np.zeros_like(vmec_dump.azmn[:, :, :, 1]))),
            ("bzmn_e", vmec_dump.bzmn[:, :, :, 0], _arr("bzmn_e", np.zeros_like(vmec_dump.bzmn[:, :, :, 0]))),
            ("bzmn_o", vmec_dump.bzmn[:, :, :, 1], _arr("bzmn_o", np.zeros_like(vmec_dump.bzmn[:, :, :, 1]))),
            ("czmn_e", vmec_dump.czmn[:, :, :, 0], _arr("czmn_e", np.zeros_like(vmec_dump.czmn[:, :, :, 0]))),
            ("czmn_o", vmec_dump.czmn[:, :, :, 1], _arr("czmn_o", np.zeros_like(vmec_dump.czmn[:, :, :, 1]))),
            ("arcon_e", vmec_dump.arcon[:, :, :, 0], _arr("arcon_e", np.zeros_like(vmec_dump.arcon[:, :, :, 0]))),
            ("arcon_o", vmec_dump.arcon[:, :, :, 1], _arr("arcon_o", np.zeros_like(vmec_dump.arcon[:, :, :, 1]))),
            ("azcon_e", vmec_dump.azcon[:, :, :, 0], _arr("azcon_e", np.zeros_like(vmec_dump.azcon[:, :, :, 0]))),
            ("azcon_o", vmec_dump.azcon[:, :, :, 1], _arr("azcon_o", np.zeros_like(vmec_dump.azcon[:, :, :, 1]))),
            ("blmn_e", vmec_dump.blmn[:, :, :, 0], _arr("blmn_e", np.zeros_like(vmec_dump.blmn[:, :, :, 0]))),
            ("blmn_o", vmec_dump.blmn[:, :, :, 1], _arr("blmn_o", np.zeros_like(vmec_dump.blmn[:, :, :, 1]))),
            ("clmn_e", vmec_dump.clmn[:, :, :, 0], _arr("clmn_e", np.zeros_like(vmec_dump.clmn[:, :, :, 0]))),
            ("clmn_o", vmec_dump.clmn[:, :, :, 1], _arr("clmn_o", np.zeros_like(vmec_dump.clmn[:, :, :, 1]))),
        ]
        for name, v, j in pairs:
            _compare(name, v, j)

        if vmec_bcovar is not None and vmec_bcovar.get("r12") is not None and vmec_bcovar.get("tau") is not None:
            bsq = vmec_bcovar["bsq"]
            r12 = vmec_bcovar["r12"]
            tau = vmec_bcovar["tau"]
            ns = int(bsq.shape[0])
            if ns < 2:
                pshalf = np.sqrt(np.maximum(np.linspace(0.0, 1.0, ns), 0.0))
            else:
                s = np.linspace(0.0, 1.0, ns)
                sh = 0.5 * (s[1:] + s[:-1])
                pshalf = np.sqrt(np.maximum(np.concatenate([sh[:1], sh], axis=0), 0.0))
            pshalf = pshalf[:, None, None]
            crmn_ref = bsq * tau * pshalf
            czmn_ref = bsq * r12

            def _compare_vmec_ref(name: str, v: np.ndarray, ref: np.ndarray) -> None:
                max_abs, max_rel, idx, vmec_v, ref_v = _max_diff_report(v, ref)
                if len(idx) == 3:
                    js, lt, lz = idx
                    loc = f"(js={js+1}, lt={lt+1}, lz={lz+1})"
                else:
                    loc = f"{idx}"
                print(
                    f"  {name:6s}: max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  at {loc}  "
                    f"vmec={vmec_v:.8e}  ref={ref_v:.8e}"
                )

            print("vmec2000 kernel vs bcovar-derived reference")
            _compare_vmec_ref("crmn_e", vmec_dump.crmn[:, :, :, 0], crmn_ref)
            _compare_vmec_ref("czmn_e", vmec_dump.czmn[:, :, :, 0], czmn_ref)


if __name__ == "__main__":
    main()

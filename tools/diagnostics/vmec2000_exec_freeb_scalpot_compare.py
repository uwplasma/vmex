#!/usr/bin/env python3
"""Compare VMEC2000 free-boundary scalpot/vacuum dumps vs vmec_jax dense path.

This tool runs:
1) VMEC2000 executable with ``VMEC_DUMP_SCALPOT=1``
2) vmec_jax ``run_fixed_boundary`` with ``VMEC_JAX_DUMP_SCALPOT=1``

and computes alignment deltas for:
- scalpot RHS vector (mode space),
- scalpot matrix (mode space),
- vacuum edge channels (where grid sizes match).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


def _parse_keyvals(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or ("=" not in s):
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_scalpot_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    mnpd2 = int(kv.get("mnpd2", "0"))
    mnpd = int(kv.get("mnpd", str(mnpd2)))
    if mnpd2 <= 0:
        raise ValueError(f"missing mnpd2 in {path}")
    section = ""
    bvec = np.zeros((mnpd2,), dtype=float)
    bvecsav = np.zeros((mnpd2,), dtype=float)
    amat_raw = np.zeros((mnpd2, mnpd2), dtype=float)
    amat = np.zeros((mnpd2, mnpd2), dtype=float)
    xmpot = np.zeros((max(0, mnpd),), dtype=np.int64)
    xnpot = np.zeros((max(0, mnpd),), dtype=np.int64)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip().lower()
            continue
        if s.startswith("#"):
            continue
        parts = s.split()
        if section == "bvec" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < mnpd2:
                bvec[i] = float(parts[1].replace("D", "E"))
        elif section == "bvecsav" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < mnpd2:
                bvecsav[i] = float(parts[1].replace("D", "E"))
        elif section == "amatrix_raw" and len(parts) >= 3:
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            if (0 <= i < mnpd2) and (0 <= j < mnpd2):
                amat_raw[i, j] = float(parts[2].replace("D", "E"))
        elif section == "amatrix_lu" and len(parts) >= 3:
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            if (0 <= i < mnpd2) and (0 <= j < mnpd2):
                amat[i, j] = float(parts[2].replace("D", "E"))
        elif section == "xmpot_xnpot" and len(parts) >= 3:
            i = int(parts[0]) - 1
            if 0 <= i < mnpd:
                xmpot[i] = int(round(float(parts[1].replace("D", "E"))))
                xnpot[i] = int(round(float(parts[2].replace("D", "E"))))
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "ivacskip": int(kv.get("ivacskip", "-1")),
        "mnpd2": mnpd2,
        "mnpd": mnpd,
        "bvec": bvec,
        "bvecsav": bvecsav,
        "amatrix_raw": amat_raw,
        "amatrix_lu": amat,
        "xmpot": xmpot,
        "xnpot": xnpot,
    }


def _mode_reindex(vmec_modes: np.ndarray, jax_modes: np.ndarray) -> np.ndarray | None:
    """Return indices selecting jax order that matches vmec order, or None."""

    if vmec_modes.shape != jax_modes.shape or vmec_modes.ndim != 2:
        return None
    key_to_idx: dict[tuple[int, int], int] = {}
    for j, key in enumerate(map(tuple, jax_modes.tolist())):
        if key not in key_to_idx:
            key_to_idx[key] = j
    out = np.empty((vmec_modes.shape[0],), dtype=np.int64)
    for i, key in enumerate(map(tuple, vmec_modes.tolist())):
        j = key_to_idx.get(key)
        if j is None:
            return None
        out[i] = int(j)
    return out


def _parse_vacuum_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    mnpd2 = int(kv.get("mnpd2", "0"))
    nuv3 = int(kv.get("nuv3", "0"))
    section = ""
    potvac = np.zeros((max(0, mnpd2),), dtype=float)
    bsqvac = np.zeros((max(0, nuv3),), dtype=float)
    bsubu = np.zeros((max(0, nuv3),), dtype=float)
    bsubv = np.zeros((max(0, nuv3),), dtype=float)
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip().lower()
            continue
        if s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        idx = int(parts[0]) - 1
        val = float(parts[1].replace("D", "E"))
        if section == "potvac" and 0 <= idx < potvac.size:
            potvac[idx] = val
        elif section == "bsqvac" and 0 <= idx < bsqvac.size:
            bsqvac[idx] = val
        elif section == "bsubu_sur" and 0 <= idx < bsubu.size:
            bsubu[idx] = val
        elif section == "bsubv_sur" and 0 <= idx < bsubv.size:
            bsubv[idx] = val
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "ivacskip": int(kv.get("ivacskip", "-1")),
        "mnpd2": mnpd2,
        "nuv3": nuv3,
        "potvac": potvac,
        "bsqvac": bsqvac,
        "bsubu_sur": bsubu,
        "bsubv_sur": bsubv,
    }


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    da = np.asarray(a, dtype=float)
    db = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(da))
    if denom <= 0.0:
        return float(np.linalg.norm(db))
    return float(np.linalg.norm(da - db) / denom)


def _rel_scaled(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    da = np.asarray(a, dtype=float).reshape(-1)
    db = np.asarray(b, dtype=float).reshape(-1)
    den = float(np.dot(db, db))
    alpha = 1.0 if den <= 0.0 else float(np.dot(da, db) / den)
    return alpha, _rel(da, alpha * db)


def _copy_input_and_mgrid(input_path: Path, workdir: Path) -> Path:
    from vmec_jax.namelist import read_indata

    src = input_path.resolve()
    dst = workdir / src.name
    shutil.copy2(src, dst)
    indata = read_indata(src)
    mg = str(indata.get("MGRID_FILE", "NONE")).strip().strip("'").strip('"')
    if mg and mg.upper() != "NONE":
        mg_src = Path(mg)
        if not mg_src.is_absolute():
            mg_src = src.parent / mg_src
        if mg_src.exists():
            mg_dst = (workdir / Path(mg).name).resolve()
            shutil.copy2(mg_src, mg_dst)
            txt = dst.read_text(encoding="utf-8")
            txt = re.sub(
                r"(?im)^\s*MGRID_FILE\s*=.*$",
                f"  MGRID_FILE = '{mg_dst}',",
                txt,
                count=1,
            )
            dst.write_text(txt, encoding="utf-8")
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--vmec-exec",
        type=Path,
        default=Path("/Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000"),
    )
    p.add_argument("--iter", type=int, default=1, help="Iteration index to compare.")
    p.add_argument("--max-iter", type=int, default=2, help="vmec_jax max_iter.")
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--json", type=Path, default=None, help="Optional output json path.")
    args = p.parse_args()

    input_path = args.input.resolve()
    vmec_exec = args.vmec_exec.resolve()
    if not input_path.exists():
        raise SystemExit(f"missing input: {input_path}")
    if not vmec_exec.exists():
        raise SystemExit(f"missing vmec executable: {vmec_exec}")

    tmp_ctx = tempfile.TemporaryDirectory(prefix="vmec_freeb_scalpot_") if args.workdir is None else None
    workdir = Path(tmp_ctx.name) if tmp_ctx is not None else args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    vmec_dump_dir = workdir / "vmec_dumps"
    jax_dump_dir = workdir / "jax_dumps"
    vmec_dump_dir.mkdir(parents=True, exist_ok=True)
    jax_dump_dir.mkdir(parents=True, exist_ok=True)
    run_input = _copy_input_and_mgrid(input_path, workdir)

    # Run VMEC2000
    env_vmec = os.environ.copy()
    env_vmec.update(
        {
            "VMEC_DUMP_SCALPOT": "1",
            "VMEC_DUMP_ITER": str(int(args.iter)),
            "VMEC_DUMP_DIR": str(vmec_dump_dir),
        }
    )
    subprocess.run(
        [str(vmec_exec), run_input.name],
        cwd=str(workdir),
        env=env_vmec,
        check=True,
        timeout=300,
    )

    # Run vmec_jax
    from vmec_jax.driver import run_fixed_boundary

    old_env = os.environ.copy()
    os.environ["VMEC_JAX_DUMP_SCALPOT"] = "1"
    os.environ["VMEC_JAX_DUMP_ITER"] = str(int(args.iter))
    os.environ["VMEC_JAX_DUMP_DIR"] = str(jax_dump_dir)
    os.environ["VMEC_JAX_FREEB_NESTOR_MODE"] = "vmec2000_like"
    os.environ["VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS"] = "1000000"
    try:
        run_fixed_boundary(
            str(run_input),
            solver="vmec2000_iter",
            max_iter=int(args.max_iter),
            multigrid=False,
            verbose=False,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    vmec_scalpot_files = sorted(vmec_dump_dir.glob(f"scalpot_iter{int(args.iter)}_ivacskip*.dat"))
    vmec_vac_files = sorted(vmec_dump_dir.glob(f"vacuum_iter{int(args.iter)}_ivacskip*.dat"))
    jax_npz = jax_dump_dir / f"scalpot_jax_iter{int(args.iter)}.npz"
    if not vmec_scalpot_files:
        raise SystemExit(f"missing VMEC scalpot dump in {vmec_dump_dir}")
    if not vmec_vac_files:
        raise SystemExit(f"missing VMEC vacuum dump in {vmec_dump_dir}")
    if not jax_npz.exists():
        raise SystemExit(f"missing vmec_jax dump: {jax_npz}")

    vmec_scal = _parse_scalpot_dump(vmec_scalpot_files[0])
    vmec_vac = _parse_vacuum_dump(vmec_vac_files[0])
    jax = dict(np.load(jax_npz, allow_pickle=False))

    vmec_modes = None
    jax_modes = None
    mode_map = None
    if ("xmpot" in vmec_scal) and ("xnpot" in vmec_scal):
        vmec_modes = np.stack([np.asarray(vmec_scal["xmpot"]), np.asarray(vmec_scal["xnpot"])], axis=1)
    if ("xmpot" in jax) and ("xnpot" in jax):
        jax_modes = np.stack([np.asarray(jax["xmpot"]), np.asarray(jax["xnpot"])], axis=1)
    if (vmec_modes is not None) and (jax_modes is not None):
        mode_map = _mode_reindex(vmec_modes, jax_modes)

    out: dict[str, Any] = {
        "workdir": str(workdir),
        "vmec_scalpot_dump": str(vmec_scalpot_files[0]),
        "vmec_vacuum_dump": str(vmec_vac_files[0]),
        "jax_dump": str(jax_npz),
        "iter": int(args.iter),
        "mode_map_applied": bool(mode_map is not None),
    }

    if "bvec_mode_sin" in jax:
        if "bvec_mode_cos" in jax:
            jax_bvec = np.concatenate([np.asarray(jax["bvec_mode_sin"]), np.asarray(jax["bvec_mode_cos"])], axis=0)
        else:
            jax_bvec = np.asarray(jax["bvec_mode_sin"])
        if mode_map is not None and mode_map.size > 0:
            mnpd = int(mode_map.size)
            if jax_bvec.size >= mnpd:
                if jax_bvec.size >= 2 * mnpd:
                    idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                    jax_bvec = jax_bvec[idx]
                else:
                    jax_bvec = jax_bvec[mode_map]
        vmec_bvec = np.asarray(vmec_scal["bvec"])
        n = min(vmec_bvec.size, jax_bvec.size)
        vm = vmec_bvec[:n]
        jj = jax_bvec[:n]
        a_bvec, rel_bvec_scaled = _rel_scaled(vm, jj)
        out["bvec"] = {
            "size_vmec": int(vmec_bvec.size),
            "size_jax": int(jax_bvec.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_bvec_scaled,
            "scale_jax_to_vmec": a_bvec,
        }

    if "amatrix_mode" in jax:
        jax_a = np.asarray(jax["amatrix_mode"], dtype=float)
        if mode_map is not None and mode_map.size > 0 and jax_a.ndim == 2:
            mnpd = int(mode_map.size)
            if jax_a.shape[0] >= mnpd and jax_a.shape[1] >= mnpd:
                if jax_a.shape[0] >= 2 * mnpd and jax_a.shape[1] >= 2 * mnpd:
                    idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                else:
                    idx = mode_map
                jax_a = jax_a[np.ix_(idx, idx)]
        vmec_a_raw = np.asarray(vmec_scal.get("amatrix_raw", np.zeros_like(vmec_scal["amatrix_lu"])), dtype=float)
        vmec_a_lu = np.asarray(vmec_scal["amatrix_lu"], dtype=float)
        vmec_a = vmec_a_raw if np.any(np.abs(vmec_a_raw) > 0.0) else vmec_a_lu
        n = min(vmec_a.shape[0], jax_a.shape[0])
        vm = vmec_a[:n, :n]
        jj = jax_a[:n, :n]
        a_mat, rel_mat_scaled = _rel_scaled(vm.reshape(-1), jj.reshape(-1))
        out["amatrix"] = {
            "shape_vmec": list(vmec_a.shape),
            "shape_jax": list(jax_a.shape),
            "shape_cmp": [int(n), int(n)],
            "rel_raw": _rel(vm.reshape(-1), jj.reshape(-1)),
            "rel_scaled": rel_mat_scaled,
            "scale_jax_to_vmec": a_mat,
            "vmec_matrix_kind": "raw" if np.any(np.abs(vmec_a_raw) > 0.0) else "lu",
        }

    if "bsqvac" in jax:
        jax_bsq = np.asarray(jax["bsqvac"], dtype=float).reshape(-1)
        vmec_bsq = np.asarray(vmec_vac["bsqvac"], dtype=float).reshape(-1)
        n = min(vmec_bsq.size, jax_bsq.size)
        vm = vmec_bsq[:n]
        jj = jax_bsq[:n]
        a_bsq, rel_bsq_scaled = _rel_scaled(vm, jj)
        out["bsqvac"] = {
            "size_vmec": int(vmec_bsq.size),
            "size_jax": int(jax_bsq.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_bsq_scaled,
            "scale_jax_to_vmec": a_bsq,
        }

    print(json.dumps(out, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

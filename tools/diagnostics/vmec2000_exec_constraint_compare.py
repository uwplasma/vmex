"""Compare VMEC2000 constraint pipeline dumps against vmec_jax.

This tool runs the VMEC2000 executable with a constraint dump enabled and
compares intermediate real-space arrays from VMEC's:

  ztemp = (rcon - rcon0)*ru0 + (zcon - zcon0)*zu0
  gcon  = alias(ztemp)

against the corresponding arrays produced by vmec_jax's constraint pipeline.
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
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


@dataclass(frozen=True)
class ConstraintDump:
    ns: int
    ntheta3: int
    nzeta: int
    gcon: np.ndarray  # (ns,ntheta3,nzeta)
    ztemp: np.ndarray
    ru0: np.ndarray
    zu0: np.ndarray
    rcon0: np.ndarray
    zcon0: np.ndarray
    rcon: np.ndarray
    zcon: np.ndarray
    tcon: np.ndarray  # (ns,)
    ard1: np.ndarray  # (ns,)
    azd1: np.ndarray  # (ns,)
    r12: np.ndarray  # (ns,ntheta3,nzeta)
    sqrtg: np.ndarray
    bsq: np.ndarray
    ru12: np.ndarray
    zu12: np.ndarray
    wint: np.ndarray


def _parse_constraints_dump(path: Path) -> ConstraintDump:
    ns = None
    ntheta3 = None
    nzeta = None
    rows: list[str] = []
    tcon_rows: list[str] = []
    ard_rows: list[str] = []
    in_tcon = False
    in_ard = False
    in_pre = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("tcon:"):
                in_tcon = True
                in_ard = False
                in_pre = False
                continue
            if line.lower().startswith("ard_azd:"):
                in_ard = True
                in_tcon = False
                in_pre = False
                continue
            if line.lower().startswith("pre:"):
                # Legacy constraints dump format: ignore the embedded `pre:` section
                # and prefer the dedicated `precond_inputs_iter*.dat` dump.
                in_pre = True
                in_tcon = False
                in_ard = False
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
            if in_pre:
                continue
            if in_tcon:
                tcon_rows.append(line)
            elif in_ard:
                ard_rows.append(line)
            else:
                rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header fields in constraints dump: {path}")

    shape = (int(ns), int(ntheta3), int(nzeta))
    def _z():
        return np.zeros(shape, dtype=float)

    gcon = _z()
    ztemp = _z()
    ru0 = _z()
    zu0 = _z()
    rcon0 = _z()
    zcon0 = _z()
    rcon = _z()
    zcon = _z()
    tcon = np.zeros((int(ns),), dtype=float)
    ard1 = np.zeros((int(ns),), dtype=float)
    azd1 = np.zeros((int(ns),), dtype=float)
    r12 = _z() * np.nan
    sqrtg = _z() * np.nan
    bsq = _z() * np.nan
    ru12 = _z() * np.nan
    zu12 = _z() * np.nan
    wint = _z() * np.nan

    for line in rows:
        toks = line.split()
        # 3 ints + 8 floats = 11 tokens.
        if len(toks) < 11:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[3:11]]
        gcon[js, lt, lz] = vals[0]
        ztemp[js, lt, lz] = vals[1]
        ru0[js, lt, lz] = vals[2]
        zu0[js, lt, lz] = vals[3]
        rcon0[js, lt, lz] = vals[4]
        zcon0[js, lt, lz] = vals[5]
        rcon[js, lt, lz] = vals[6]
        zcon[js, lt, lz] = vals[7]

    for line in tcon_rows:
        toks = line.split()
        if len(toks) < 2:
            continue
        js = int(toks[0]) - 1
        if 0 <= js < tcon.shape[0]:
            tcon[js] = float(toks[1].replace("D", "E").replace("d", "E"))

    for line in ard_rows:
        toks = line.split()
        if len(toks) < 3:
            continue
        js = int(toks[0]) - 1
        if 0 <= js < ard1.shape[0]:
            ard1[js] = float(toks[1].replace("D", "E").replace("d", "E"))
            azd1[js] = float(toks[2].replace("D", "E").replace("d", "E"))

    return ConstraintDump(
        ns=int(ns),
        ntheta3=int(ntheta3),
        nzeta=int(nzeta),
        gcon=gcon,
        ztemp=ztemp,
        ru0=ru0,
        zu0=zu0,
        rcon0=rcon0,
        zcon0=zcon0,
        rcon=rcon,
        zcon=zcon,
        tcon=tcon,
        ard1=ard1,
        azd1=azd1,
        r12=r12,
        sqrtg=sqrtg,
        bsq=bsq,
        ru12=ru12,
        zu12=zu12,
        wint=wint,
    )


def _parse_precond_inputs_dump(
    path: Path,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            if line.startswith("r12") or line.startswith("ru12"):
                continue
            rows.append(line)
    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Missing header fields in precond dump: {path}")

    shape = (int(ns), int(ntheta3), int(nzeta))

    def _z():
        return np.zeros(shape, dtype=float)

    r12 = _z()
    sqrtg = _z()
    bsq = _z()
    ru12 = _z()
    zu12 = _z()
    wint = _z()

    for line in rows:
        toks = line.split()
        # 3 ints + 6 floats = 9 tokens.
        if len(toks) < 9:
            continue
        js = int(toks[0]) - 1
        lt = int(toks[1]) - 1
        lz = int(toks[2]) - 1
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[3:9]]
        r12[js, lt, lz] = vals[0]
        sqrtg[js, lt, lz] = vals[1]
        bsq[js, lt, lz] = vals[2]
        ru12[js, lt, lz] = vals[3]
        zu12[js, lt, lz] = vals[4]
        wint[js, lt, lz] = vals[5]

    return int(ns), int(ntheta3), int(nzeta), r12, sqrtg, bsq, ru12, zu12, wint


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

    with tempfile.TemporaryDirectory(prefix="vmec2000_constraints_") as td:
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
            "VMEC_DUMP_CONSTRAINTS": "1",
            "VMEC_DUMP_PRECOND": "1",
            "VMEC_DUMP_ITER": str(int(args.iter)),
            "VMEC_DUMP_DIR": str(workdir),
        }
        cmd = [str(vmec2000_exe), input_local.name]
        subprocess.run(cmd, cwd=workdir, env={**os.environ, **env}, check=False, capture_output=True, text=True)

        vmec_dump_path = workdir / f"constraints_iter{int(args.iter)}.dat"
        if not vmec_dump_path.exists():
            raise SystemExit(f"VMEC2000 constraints dump not found: {vmec_dump_path}")
        vmec_dump = _parse_constraints_dump(vmec_dump_path)
        precond_path = workdir / f"precond_inputs_iter{int(args.iter)}.dat"
        if precond_path.exists():
            ns, ntheta3, nzeta, r12, sqrtg, bsq, ru12, zu12, wint = _parse_precond_inputs_dump(precond_path)
            if (ns, ntheta3, nzeta) != (vmec_dump.ns, vmec_dump.ntheta3, vmec_dump.nzeta):
                raise SystemExit(
                    f"Precond dump header mismatch: {(ns,ntheta3,nzeta)} vs constraints {(vmec_dump.ns,vmec_dump.ntheta3,vmec_dump.nzeta)}"
                )
            vmec_dump = replace(
                vmec_dump,
                r12=r12,
                sqrtg=sqrtg,
                bsq=bsq,
                ru12=ru12,
                zu12=zu12,
                wint=wint,
            )

        # Run vmec_jax with constraint dump enabled.
        jax_dump_dir = workdir / "jax"
        jax_dump_dir.mkdir(parents=True, exist_ok=True)
        with _env(
            {
                "VMEC_JAX_DUMP_CONSTRAINTS": "1",
                "VMEC_JAX_DUMP_BCOVAR": "1",
                "VMEC_JAX_DUMP_ITER": str(int(args.iter)),
                "VMEC_JAX_DUMP_DIR": str(jax_dump_dir),
            }
        ):
            run = vj.run_fixed_boundary(
                input_path,
                solver="vmec2000_iter",
                max_iter=int(args.iter),
                multigrid_use_input_niter=bool(args.use_input_niter),
                ns_override=int(args.single_ns) if args.single_ns is not None else None,
                verbose=False,
            )

        jax_dump_path = jax_dump_dir / f"constraints_raw_iter{int(args.iter)}.npz"
        if not jax_dump_path.exists():
            raise SystemExit(f"vmec_jax constraints dump not found: {jax_dump_path}")
        jax_dump = np.load(jax_dump_path)

        jax_bcovar_path = jax_dump_dir / f"bcovar_raw_iter{int(args.iter)}.npz"
        jax_bcovar = None
        if jax_bcovar_path.exists():
            jax_bcovar = np.load(jax_bcovar_path)

        def _arr(name: str) -> np.ndarray:
            return np.asarray(jax_dump[name], dtype=float)

        print("constraint pipeline comparison (vmec2000 vs vmec_jax)")
        for name in ("tcon", "gcon", "ztemp", "ru0", "zu0", "rcon0", "zcon0", "rcon", "zcon"):
            v = getattr(vmec_dump, name)
            if name not in jax_dump:
                print(f"  {name:6s}: skipped (missing in vmec_jax dump)")
                continue
            j = _arr(name)
            max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
            if len(idx) == 1:
                (js,) = idx
                loc = f"(js={js+1})"
            else:
                js, lt, lz = idx
                loc = f"(js={js+1}, lt={lt+1}, lz={lz+1})"
            print(
                f"  {name:6s}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} at {loc} "
                f"vmec={vmec_v:.8e} jax={jax_v:.8e}"
            )

        if (jax_bcovar is not None) and np.all(np.isfinite(vmec_dump.wint)):
            # Compare VMEC `precondn` inputs and diagonal `ard/azd` outputs against the
            # values produced inside vmec_jax on the same iteration.
            s = np.asarray(run.static.s, dtype=float)
            hs = float(s[1] - s[0]) if s.size >= 2 else 0.0
            ohs = 0.0 if hs == 0.0 else 1.0 / hs

            r12 = np.asarray(jax_bcovar["r12"], dtype=float)
            sqrtg = np.asarray(jax_bcovar["sqrtg"], dtype=float)
            bsq = np.asarray(jax_bcovar["bsq"], dtype=float)
            ru12 = np.asarray(jax_bcovar["ru12"], dtype=float)
            zu12 = np.asarray(jax_bcovar["zu12"], dtype=float)
            wint3 = np.asarray(jax_bcovar["wint"], dtype=float)

            print("precondn input parity (VMEC2000 vs vmec_jax)")
            for name, v, j in (
                ("r12", vmec_dump.r12, r12),
                ("sqrtg", vmec_dump.sqrtg, sqrtg),
                ("bsq", vmec_dump.bsq, bsq),
                ("ru12", vmec_dump.ru12, ru12),
                ("zu12", vmec_dump.zu12, zu12),
                ("wint", vmec_dump.wint, wint3),
            ):
                max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
                js, lt, lz = idx
                loc = f"(js={js+1}, lt={lt+1}, lz={lz+1})"
                print(
                    f"  {name:6s}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} at {loc} "
                    f"vmec={vmec_v:.8e} jax={jax_v:.8e}"
                )

            # Recompute the reduced precondn diagonal and compare against VMEC's dumped ard/azd.
            # VMEC2000 precondn uses `pfactor = -4*r0scale**2` (v8.51+). With
            # VMEC's normalization (`mscale(0)=nscale(0)=1`), `r0scale=1`.
            pfactor = -4.0
            gs = np.where(sqrtg != 0.0, sqrtg, 1.0)
            ptau = (pfactor * (r12 * r12) * bsq * wint3) / gs
            ax_r = np.sum(ptau * ((zu12 * ohs) ** 2), axis=(1, 2))
            ax_z = np.sum(ptau * ((ru12 * ohs) ** 2), axis=(1, 2))
            if ax_r.size:
                ax_r[0] = 0.0
                ax_z[0] = 0.0
            ard1 = ax_r + np.concatenate([ax_r[1:], np.zeros((1,), dtype=float)], axis=0)
            azd1 = ax_z + np.concatenate([ax_z[1:], np.zeros((1,), dtype=float)], axis=0)

            print("precondn diag comparison (VMEC2000 ard/azd vs vmec_jax recompute)")
            for name, v, j in (("ard1", vmec_dump.ard1, ard1), ("azd1", vmec_dump.azd1, azd1)):
                max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(v, j)
                (js,) = idx
                loc = f"(js={js+1})"
                print(
                    f"  {name:6s}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} at {loc} "
                    f"vmec={vmec_v:.8e} jax={jax_v:.8e}"
                )
        else:
            print("precondn input/diag parity skipped (missing bcovar dump or VMEC precond inputs).")


if __name__ == "__main__":
    main()

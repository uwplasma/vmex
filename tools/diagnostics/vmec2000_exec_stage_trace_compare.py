"""Compare VMEC2000 *executable* traces to vmec_jax multigrid stages (axisym-friendly).

This diagnostic is intended for parity debugging of the fixed-boundary iteration
loop, without relying on the `vmec` Python extension.

It runs:
  1) `xvmec2000 input.*` (STELLOPT/VMEC2000 build) in a temp workdir and parses
     the printed iteration table (fsqr/fsqz/fsql at selected iterations).
  2) `vmec_jax.run_fixed_boundary(..., solver="vmec2000_iter")` and compares the
     corresponding stage/iteration residual scalars.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vmec_jax.api as vj
from vmec_jax.wout import read_wout


@dataclass(frozen=True)
class Vmec2000PrintedRow:
    it: int
    fsqr: float
    fsqz: float
    fsql: float


@dataclass(frozen=True)
class Vmec2000PrintedStage:
    ns: int
    niter: int
    ftolv: float
    rows: list[Vmec2000PrintedRow]


@dataclass(frozen=True)
class Vmec2000Threed1Row:
    """One row of the threed1 force-iteration table."""

    it: int
    fsqr: float
    fsqz: float
    fsql: float
    fsqr1: float
    fsqz1: float
    fsql1: float
    delt0r: float | None = None
    r00: float | None = None
    w: float | None = None


@dataclass(frozen=True)
class Vmec2000Threed1Stage:
    ns: int
    niter: int
    ftolv: float
    rows: list[Vmec2000Threed1Row]


_RE_STAGE = re.compile(
    r"^\s*NS\s*=\s*(\d+)\s+NO\.\s+FOURIER\s+MODES\s*=\s*(\d+)\s+FTOLV\s*=\s*([0-9.Ee+-]+)\s+NITER\s*=\s*(\d+)"
)
_RE_ROW = re.compile(r"^\s*(\d+)\s+([0-9.DdEe+-]+)\s+([0-9.DdEe+-]+)\s+([0-9.DdEe+-]+)\s+")


def _parse_vmec2000_stdout(text: str) -> list[Vmec2000PrintedStage]:
    stages: list[Vmec2000PrintedStage] = []
    current: Vmec2000PrintedStage | None = None
    rows: list[Vmec2000PrintedRow] = []

    def _flush():
        nonlocal current, rows
        if current is None:
            return
        stages.append(Vmec2000PrintedStage(ns=current.ns, niter=current.niter, ftolv=current.ftolv, rows=rows))
        current = None
        rows = []

    for line in text.splitlines():
        m = _RE_STAGE.match(line)
        if m:
            _flush()
            ns = int(m.group(1))
            ftolv = float(m.group(3).replace("D", "E").replace("d", "E"))
            niter = int(m.group(4))
            current = Vmec2000PrintedStage(ns=ns, niter=niter, ftolv=ftolv, rows=[])
            continue
        m = _RE_ROW.match(line)
        if m and current is not None:
            it = int(m.group(1))
            fsqr = float(m.group(2).replace("D", "E").replace("d", "E"))
            fsqz = float(m.group(3).replace("D", "E").replace("d", "E"))
            fsql = float(m.group(4).replace("D", "E").replace("d", "E"))
            rows.append(Vmec2000PrintedRow(it=it, fsqr=fsqr, fsqz=fsqz, fsql=fsql))
    _flush()
    return stages


def _parse_vmec2000_threed1(path: Path) -> list[Vmec2000Threed1Stage]:
    """Parse VMEC2000 `threed1.*` stage headers + per-iteration tables."""
    text = path.read_text()
    stages: list[Vmec2000Threed1Stage] = []
    current: Vmec2000Threed1Stage | None = None
    rows: list[Vmec2000Threed1Row] = []
    in_table = False

    def _flush() -> None:
        nonlocal current, rows, in_table
        if current is None:
            return
        stages.append(Vmec2000Threed1Stage(ns=current.ns, niter=current.niter, ftolv=current.ftolv, rows=rows))
        current = None
        rows = []
        in_table = False

    def _f(tok: str) -> float:
        return float(tok.replace("D", "E").replace("d", "E"))

    for line in text.splitlines():
        m = _RE_STAGE.match(line)
        if m:
            _flush()
            ns = int(m.group(1))
            ftolv = float(m.group(3).replace("D", "E").replace("d", "E"))
            niter = int(m.group(4))
            current = Vmec2000Threed1Stage(ns=ns, niter=niter, ftolv=ftolv, rows=[])
            continue

        if current is None:
            continue

        if line.strip().startswith("ITER") and ("FSQR" in line) and ("fsqr" in line):
            in_table = True
            continue
        if not in_table:
            continue
        if line.lstrip().startswith("MHD Energy"):
            in_table = False
            continue

        toks = line.split()
        if len(toks) < 8 or (not toks[0].isdigit()):
            continue
        it = int(toks[0])

        # Typical format:
        #   ITER FSQR FSQZ FSQL fsqr fsqz fsql DELT RAX WMHD BETA <M>
        #
        # In `printout.f` this corresponds to:
        #   (fsqr, fsqz, fsql, fsqr1, fsqz1, fsql1, delt0r, r00, w, betav, avm)
        # i.e. the lowercase `fsq*` headers are the *preconditioned* scalars.
        r = Vmec2000Threed1Row(
            it=it,
            fsqr=_f(toks[1]),
            fsqz=_f(toks[2]),
            fsql=_f(toks[3]),
            fsqr1=_f(toks[4]),
            fsqz1=_f(toks[5]),
            fsql1=_f(toks[6]),
            delt0r=_f(toks[7]) if len(toks) > 7 else None,
            r00=_f(toks[8]) if len(toks) > 8 else None,
            w=_f(toks[9]) if len(toks) > 9 else None,
        )
        rows.append(r)

    _flush()
    return stages


def _rel_rms(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-16) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    num = float(np.sqrt(np.mean((x - y) ** 2)))
    den = float(np.sqrt(np.mean(y**2)))
    return num / max(eps, den)


def _max_abs_rel_err(vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1e-30) -> tuple[float, float, int]:
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan"), -1
    mask = np.isfinite(diff)
    if not bool(np.any(mask)):
        return float("nan"), float("nan"), -1
    i = int(np.argmax(np.where(mask, diff, -np.inf)))
    max_abs = float(diff[i])
    denom = max(eps, float(abs(vmec_vals[i])))
    max_rel = float(max_abs / denom)
    return max_abs, max_rel, i


def _patch_indata(text: str, *, updates: dict[str, str]) -> str:
    """Patch simple `&INDATA` assignments in a VMEC namelist.

    This is intentionally minimal: it replaces (or inserts) key/value assignments
    in the `&INDATA` block so diagnostics can force e.g. `NSTEP=1` and short
    iteration counts.
    """
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

    # Insert missing assignments just before the "/" terminator.
    insert_lines = []
    for k_up, v in updates.items():
        if not found[k_up.upper()]:
            insert_lines.append(f"  {k_up.upper()} = {v}")
    if insert_lines:
        lines = lines[:end_idx] + insert_lines + lines[end_idx:]
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _distribute_iters(*, iters: int, nstep: int) -> list[int]:
    iters = int(iters)
    nstep = int(nstep)
    if iters <= 0:
        return [0]
    if nstep <= 1:
        return [iters]
    base, rem = divmod(iters, nstep)
    if base == 0:
        return [iters]
    return [base + (1 if i < rem else 0) for i in range(nstep)]


def _resolve_stage_controls(*, cfg, indata, max_iter: int, use_input_niter: bool) -> tuple[list[int], list[int], list[float]]:
    ns_array = indata.get("NS_ARRAY", None)
    if isinstance(ns_array, list) and ns_array:
        ns_stages = [int(v) for v in ns_array]
    else:
        ns_stages = [int(getattr(cfg, "ns", 0)) or int(indata.get_int("NS", 0)) or 0]
    ns_stages = [int(v) for v in ns_stages if int(v) > 0]
    if not ns_stages:
        raise ValueError("Failed to resolve NS_ARRAY stages for VMEC2000 parity run.")

    nstep = len(ns_stages)
    ftol_default = float(indata.get_float("FTOL", 1e-10))

    if use_input_niter:
        niter_array = indata.get("NITER_ARRAY", None)
        ftol_array = indata.get("FTOL_ARRAY", None)
        niter_stages = (
            [int(v) for v in niter_array] if isinstance(niter_array, list) and len(niter_array) == nstep else None
        )
        ftol_stages = (
            [float(v) for v in ftol_array] if isinstance(ftol_array, list) and len(ftol_array) == nstep else None
        )
        if niter_stages is None:
            niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
        else:
            budget = int(max_iter)
            if budget < nstep:
                # Too few iterations to stage; collapse to the final grid.
                ns_stages = [int(ns_stages[-1])]
                nstep = 1
                niter_stages = [int(max(budget, 1))]
                if ftol_stages is not None:
                    ftol_stages = [float(ftol_stages[-1])]
            else:
                base = [1] * nstep
                remaining = budget - nstep
                caps = [max(0, int(n) - 1) for n in niter_stages]
                out = base[:]
                for i in range(nstep - 1, -1, -1):
                    if remaining <= 0:
                        break
                    take = min(caps[i], remaining)
                    out[i] += take
                    remaining -= take
                if remaining > 0:
                    out[-1] += remaining
                niter_stages = out
        if ftol_stages is None:
            ftol_stages = [ftol_default] * nstep
    else:
        niter_stages = _distribute_iters(iters=int(max_iter), nstep=int(nstep))
        ftol_stages = [ftol_default] * nstep

    nrun = min(len(ns_stages), len(niter_stages), len(ftol_stages))
    return ns_stages[:nrun], niter_stages[:nrun], ftol_stages[:nrun]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="circular_tokamak")
    p.add_argument("--input", type=str, default=None, help="Path to input.* (overrides --case).")
    p.add_argument("--vmec2000", type=str, default=None, help="Path to xvmec2000 executable.")
    p.add_argument("--max-iter", type=int, default=2, help="Total iteration budget for vmec_jax.")
    p.add_argument(
        "--vmec-nstep",
        type=int,
        default=1,
        help="Override VMEC2000 `NSTEP` (printout cadence). Use 1 for per-iteration threed1 traces.",
    )
    p.add_argument(
        "--single-ns",
        type=int,
        default=None,
        help="If set, force both VMEC2000 and vmec_jax to run a single grid at this ns (no multigrid).",
    )
    p.add_argument(
        "--use-input-niter",
        action="store_true",
        help="Use VMEC input NITER_ARRAY/FTOL_ARRAY staging (still capped by --max-iter).",
    )
    p.add_argument(
        "--delt-source",
        choices=("time_step", "dt_eff"),
        default="time_step",
        help="Which vmec_jax series to compare against VMEC2000 DELT.",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for fail-fast mismatch detection.",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for fail-fast mismatch detection.",
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero at the first mismatch beyond tolerances (default: True).",
    )
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

    # Load indata once so we can reuse `FTOL` etc for diagnostic patches.
    _cfg_in, _indata_in = vj.load_input(input_path)
    ftol_default = float(_indata_in.get_float("FTOL", 1e-10))

    # Resolve stage controls to match vmec_jax staging.
    ns_stages_eff: list[int] | None = None
    niter_stages_eff: list[int] | None = None
    ftol_stages_eff: list[float] | None = None
    if args.single_ns is None:
        ns_stages_eff, niter_stages_eff, ftol_stages_eff = _resolve_stage_controls(
            cfg=_cfg_in,
            indata=_indata_in,
            max_iter=int(args.max_iter),
            use_input_niter=bool(args.use_input_niter),
        )

    # --- Run VMEC2000 executable in an isolated workdir ---
    threed1_stages: list[Vmec2000Threed1Stage] | None = None
    with tempfile.TemporaryDirectory(prefix="vmec2000_exec_") as td:
        workdir = Path(td)
        input_local = workdir / input_path.name
        shutil.copy2(input_path, input_local)

        # Force per-iteration printout cadence by patching `NSTEP`.
        indata_text = input_local.read_text()
        updates = {"NSTEP": str(int(args.vmec_nstep))}

        # Optional single-grid debug mode for tighter iteration parity.
        if args.single_ns is not None:
            ns = int(args.single_ns)
            updates |= {
                "NS_ARRAY": f"{ns}",
                "NITER_ARRAY": f"{int(args.max_iter)}",
                "FTOL_ARRAY": f"{ftol_default:.16e}",
                "NITER": f"{int(args.max_iter)}",
            }
        elif ns_stages_eff and niter_stages_eff and ftol_stages_eff:
            updates |= {
                "NS_ARRAY": "  ".join(str(int(v)) for v in ns_stages_eff),
                "NITER_ARRAY": "  ".join(str(int(v)) for v in niter_stages_eff),
                "FTOL_ARRAY": "  ".join(f"{float(v):.16e}" for v in ftol_stages_eff),
                "NITER": f"{int(sum(niter_stages_eff))}",
            }

        input_local.write_text(_patch_indata(indata_text, updates=updates))
        cmd = [str(vmec2000_exe), input_local.name]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, check=False)
        stdout = proc.stdout + "\n" + proc.stderr

        stages = _parse_vmec2000_stdout(stdout)
        if not stages:
            raise SystemExit("Failed to parse VMEC2000 stdout (no stages found).")

        # Prefer parsing `threed1.*` when available: it contains both physical
        # (FSQR/FSQZ/FSQL) and preconditioned (fsqr/fsqz/fsql) scalars plus DELT.
        suffix = input_path.name.split("input.", 1)[-1]
        threed1_path = workdir / f"threed1.{suffix}"
        if not threed1_path.exists():
            # Fallback: pick the first threed1.* in the workdir.
            cands = sorted(workdir.glob("threed1.*"))
            threed1_path = cands[0] if cands else threed1_path
        if threed1_path.exists():
            try:
                threed1_stages = _parse_vmec2000_threed1(threed1_path)
            except Exception:
                threed1_stages = None

        # Read the VMEC2000 wout for end-state comparison when present.
        wout_name = "wout_" + input_path.name.split("input.", 1)[-1] + ".nc"
        wout_path = workdir / wout_name
        wout = read_wout(wout_path) if wout_path.exists() else None

        # --- Run vmec_jax with VMEC-style multigrid staging ---
        run = vj.run_fixed_boundary(
            input_path,
            solver="vmec2000_iter",
            max_iter=int(args.max_iter),
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=False,
            ns_override=int(args.single_ns) if args.single_ns is not None else None,
        )

    # --- Report ---
    use_threed1 = bool(threed1_stages)
    vmec_stages = threed1_stages if use_threed1 else stages

    print()
    print("VMEC2000 stages:")
    if use_threed1:
        print("  source: threed1.* (physical + preconditioned + DELT)")
    else:
        print("  source: stdout (preconditioned only)")
    for i, st in enumerate(vmec_stages):
        its = [r.it for r in st.rows]
        it_str = ", ".join(str(v) for v in its) if its else "(no rows)"
        print(f"  stage {i+1}: ns={st.ns} niter={st.niter} ftolv={st.ftolv:.2e} printed iters: {it_str}")

    diag = getattr(run.result, "diagnostics", {}) if run.result is not None else {}
    offsets = np.asarray(diag.get("multigrid_stage_offsets", np.zeros((0,), dtype=int)), dtype=int)
    ns_stages = np.asarray(diag.get("multigrid_ns_stages", np.zeros((0,), dtype=int)), dtype=int)

    fsqr = np.asarray(getattr(run.result, "fsqr2_history", np.zeros((0,), dtype=float)), dtype=float)
    fsqz = np.asarray(getattr(run.result, "fsqz2_history", np.zeros((0,), dtype=float)), dtype=float)
    fsql = np.asarray(getattr(run.result, "fsql2_history", np.zeros((0,), dtype=float)), dtype=float)
    fsqr1 = np.asarray(diag.get("fsqr1_history", np.zeros((0,), dtype=float)), dtype=float)
    fsqz1 = np.asarray(diag.get("fsqz1_history", np.zeros((0,), dtype=float)), dtype=float)
    fsql1 = np.asarray(diag.get("fsql1_history", np.zeros((0,), dtype=float)), dtype=float)
    if args.delt_source == "dt_eff":
        delt = np.asarray(diag.get("dt_eff_history", np.zeros((0,), dtype=float)), dtype=float)
    else:
        delt = np.asarray(diag.get("time_step_history", np.zeros((0,), dtype=float)), dtype=float)
    r00 = np.asarray(diag.get("r00_history", np.zeros((0,), dtype=float)), dtype=float)
    w = np.asarray(diag.get("w_vmec_history", np.zeros((0,), dtype=float)), dtype=float)

    print()
    if use_threed1:
        print("Stage/iter comparison (VMEC2000 threed1 vs vmec_jax histories):")
        print(
            "  stage  it    fsqr(vmec)   fsqr(jax)    fsqz(vmec)   fsqz(jax)    fsql(vmec)   fsql(jax)  "
            "  fsqr1(vmec)  fsqr1(jax)   fsqz1(vmec)  fsqz1(jax)   fsql1(vmec)  fsql1(jax)   "
            "  delt0r(vmec) delt0r(jax)   r00(vmec)     r00(jax)        w(vmec)       w(jax)"
        )
    else:
        print("Stage/iter comparison (VMEC2000 stdout rows vs vmec_jax histories):")
        print("  stage  it    fsqr(vmec)   fsqr(jax)    fsqz(vmec)   fsqz(jax)    fsql(vmec)   fsql(jax)")

    # Collect matched-row values for a summary diff report.
    diff_rows: list[tuple[int, int]] = []  # (stage, iter)
    diff_cols_vmec: dict[str, list[float]] = {}
    diff_cols_jax: dict[str, list[float]] = {}
    if use_threed1:
        for name in ("fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1", "delt0r", "r00", "w"):
            diff_cols_vmec[name] = []
            diff_cols_jax[name] = []

    def _matches(vmec_val: float, jax_val: float) -> bool:
        if not (np.isfinite(vmec_val) and np.isfinite(jax_val)):
            return False
        return abs(vmec_val - jax_val) <= max(float(args.atol), float(args.rtol) * abs(vmec_val))

    # Stage transition parity (ns + offsets).
    if offsets.size and ns_stages.size:
        vmec_ns = np.asarray([int(st.ns) for st in vmec_stages], dtype=int)
        vmec_niter = np.asarray([int(st.niter) for st in vmec_stages], dtype=int)
        vmec_offsets = np.concatenate([[0], np.cumsum(vmec_niter[:-1])]).astype(int) if vmec_niter.size else np.zeros((0,), dtype=int)

        ns_ok = bool(vmec_ns.size == ns_stages.size) and bool(np.all(vmec_ns == ns_stages[: vmec_ns.size]))
        off_ok = bool(vmec_offsets.size == offsets.size) and bool(np.all(vmec_offsets == offsets[: vmec_offsets.size]))
        if not ns_ok or not off_ok:
            print()
            print("Stage transition mismatch:")
            if not ns_ok:
                print(f"  vmec ns_stages={vmec_ns.tolist()}  jax ns_stages={ns_stages.tolist()}")
            if not off_ok:
                print(f"  vmec offsets={vmec_offsets.tolist()}  jax offsets={offsets.tolist()}")
            if bool(args.fail_fast):
                raise SystemExit(2)

    for stage_i, st in enumerate(vmec_stages):
        if stage_i >= offsets.size or stage_i >= ns_stages.size:
            continue
        off = int(offsets[stage_i])
        for row in st.rows:
            j = off + max(int(row.it) - 1, 0)
            if j < 0 or j >= max(fsqr.size, fsqr1.size, delt.size, r00.size, w.size):
                continue
            if use_threed1:
                assert isinstance(row, Vmec2000Threed1Row)
                print(
                    f"  {stage_i+1:>3d} {row.it:>4d}  "
                    f"{row.fsqr:>11.3e} {fsqr[j] if j < fsqr.size else float('nan'):>11.3e}  "
                    f"{row.fsqz:>11.3e} {fsqz[j] if j < fsqz.size else float('nan'):>11.3e}  "
                    f"{row.fsql:>11.3e} {fsql[j] if j < fsql.size else float('nan'):>11.3e}  "
                    f"{row.fsqr1:>11.3e} {fsqr1[j] if j < fsqr1.size else float('nan'):>11.3e}  "
                    f"{row.fsqz1:>11.3e} {fsqz1[j] if j < fsqz1.size else float('nan'):>11.3e}  "
                    f"{row.fsql1:>11.3e} {fsql1[j] if j < fsql1.size else float('nan'):>11.3e}  "
                    f"{(row.delt0r if row.delt0r is not None else float('nan')):>11.3e} {delt[j] if j < delt.size else float('nan'):>11.3e}  "
                    f"{(row.r00 if row.r00 is not None else float('nan')):>11.3e} {r00[j] if j < r00.size else float('nan'):>11.3e}  "
                    f"{(row.w if row.w is not None else float('nan')):>11.3e} {w[j] if j < w.size else float('nan'):>11.3e}"
                )
                diff_rows.append((int(stage_i + 1), int(row.it)))
                diff_cols_vmec["fsqr"].append(float(row.fsqr))
                diff_cols_jax["fsqr"].append(float(fsqr[j]))
                diff_cols_vmec["fsqz"].append(float(row.fsqz))
                diff_cols_jax["fsqz"].append(float(fsqz[j]))
                diff_cols_vmec["fsql"].append(float(row.fsql))
                diff_cols_jax["fsql"].append(float(fsql[j]))
                diff_cols_vmec["fsqr1"].append(float(row.fsqr1))
                diff_cols_jax["fsqr1"].append(float(fsqr1[j] if j < fsqr1.size else float("nan")))
                diff_cols_vmec["fsqz1"].append(float(row.fsqz1))
                diff_cols_jax["fsqz1"].append(float(fsqz1[j] if j < fsqz1.size else float("nan")))
                diff_cols_vmec["fsql1"].append(float(row.fsql1))
                diff_cols_jax["fsql1"].append(float(fsql1[j] if j < fsql1.size else float("nan")))
                diff_cols_vmec["delt0r"].append(float(row.delt0r if row.delt0r is not None else float("nan")))
                diff_cols_jax["delt0r"].append(float(delt[j] if j < delt.size else float("nan")))
                diff_cols_vmec["r00"].append(float(row.r00 if row.r00 is not None else float("nan")))
                diff_cols_jax["r00"].append(float(r00[j] if j < r00.size else float("nan")))
                diff_cols_vmec["w"].append(float(row.w if row.w is not None else float("nan")))
                diff_cols_jax["w"].append(float(w[j] if j < w.size else float("nan")))

                if bool(args.fail_fast):
                    pairs = [
                        ("fsqr", float(row.fsqr), float(fsqr[j] if j < fsqr.size else float("nan"))),
                        ("fsqz", float(row.fsqz), float(fsqz[j] if j < fsqz.size else float("nan"))),
                        ("fsql", float(row.fsql), float(fsql[j] if j < fsql.size else float("nan"))),
                        ("fsqr1", float(row.fsqr1), float(fsqr1[j] if j < fsqr1.size else float("nan"))),
                        ("fsqz1", float(row.fsqz1), float(fsqz1[j] if j < fsqz1.size else float("nan"))),
                        ("fsql1", float(row.fsql1), float(fsql1[j] if j < fsql1.size else float("nan"))),
                        ("delt0r", float(row.delt0r if row.delt0r is not None else float("nan")), float(delt[j] if j < delt.size else float("nan"))),
                        ("r00", float(row.r00 if row.r00 is not None else float("nan")), float(r00[j] if j < r00.size else float("nan"))),
                        ("wmhd", float(row.w if row.w is not None else float("nan")), float(w[j] if j < w.size else float("nan"))),
                    ]
                    for name, v, jv in pairs:
                        if not _matches(v, jv):
                            print()
                            print("First mismatch beyond tolerance:")
                            print(f"  stage={stage_i+1} iter={row.it} field={name}")
                            print(f"  vmec2000={v:.6e}  vmec_jax={jv:.6e}")
                            print(f"  tol: rtol={args.rtol:.3e} atol={args.atol:.3e}")
                            raise SystemExit(2)
            else:
                assert isinstance(row, Vmec2000PrintedRow)
                print(
                    f"  {stage_i+1:>3d} {row.it:>4d}  "
                    f"{row.fsqr:>11.3e} {fsqr[j]:>11.3e}  "
                    f"{row.fsqz:>11.3e} {fsqz[j]:>11.3e}  "
                    f"{row.fsql:>11.3e} {fsql[j]:>11.3e}"
                )

    if use_threed1 and diff_rows:
        print()
        print("Diff summary (max abs / max rel vs VMEC2000 threed1):")
        for name in ("fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1", "delt0r", "r00", "w"):
            v = np.asarray(diff_cols_vmec[name], dtype=float)
            jv = np.asarray(diff_cols_jax[name], dtype=float)
            max_abs, max_rel, idx = _max_abs_rel_err(v, jv)
            if idx >= 0:
                st_i, it_i = diff_rows[idx]
                where = f"(stage={st_i}, iter={it_i})"
            else:
                where = ""
            print(f"  {name:>6s}: {max_abs:>11.3e} / {max_rel:>11.3e}  {where}")

    if wout is not None:
        rmnc_err = _rel_rms(np.asarray(run.state.Rcos), np.asarray(wout.rmnc))
        zmns_err = _rel_rms(np.asarray(run.state.Zsin), np.asarray(wout.zmns))
        fsq_ref = float(wout.fsqr + wout.fsqz + wout.fsql)
        fsqr_new, fsqz_new, fsql_new = vj.residual_scalars_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            use_vmec_synthesis=True,
        )
        fsq_new = float(fsqr_new + fsqz_new + fsql_new)
        print()
        print("End-state comparison vs VMEC2000 wout:")
        print(f"  fsq_total: vmec={fsq_ref:.3e}  jax={fsq_new:.3e}")
        print(f"  rmnc relRMS={rmnc_err:.3e}  zmns relRMS={zmns_err:.3e}")


if __name__ == "__main__":
    main()

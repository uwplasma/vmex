"""Helpers for running the VMEC2000 executable and parsing per-iteration traces."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Vmec2000Threed1Row:
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


@dataclass(frozen=True)
class Vmec2000ExecResult:
    workdir: Path
    input_path: Path
    returncode: int
    stdout: str
    stderr: str
    runtime_s: float
    threed1_path: Path | None
    stages: list[Vmec2000Threed1Stage]


_RE_STAGE = re.compile(
    r"^\s*NS\s*=\s*(\d+)\s+NO\.\s+FOURIER\s+MODES\s*=\s*(\d+)\s+FTOLV\s*=\s*([0-9.DdEe+-]+)\s+NITER\s*=\s*([+-]?\d+)"
)
_RE_MGRID_FILE = re.compile(r"^\s*MGRID_FILE\s*=\s*['\"]?([^'\",\s]+)['\"]?", flags=re.IGNORECASE | re.MULTILINE)


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
            if niter < 0:
                current = None
                in_table = False
                continue
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


def _patch_indata(text: str, *, updates: dict[str, str]) -> str:
    """Patch simple `&INDATA` assignments in a VMEC namelist."""
    lines = text.splitlines()
    in_block = False
    end_idx = None
    found = {k.upper(): False for k in updates}

    key_re = {k.upper(): re.compile(rf"^(\s*){re.escape(k)}\s*=", flags=re.IGNORECASE) for k in updates}
    any_assignment_re = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^)]*\))?\s*=")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.upper().startswith("&INDATA"):
            in_block = True
            i += 1
            continue
        if in_block and stripped.startswith("/"):
            end_idx = i
            break
        if not in_block:
            i += 1
            continue
        for key, regex in key_re.items():
            m = regex.match(line)
            if m:
                indent = m.group(1) or ""
                lines[i] = f"{indent}{key} = {updates[key]}"
                found[key] = True
                j = i + 1
                while j < len(lines):
                    next_stripped = lines[j].strip()
                    if next_stripped.startswith("/") or any_assignment_re.match(lines[j]):
                        break
                    if next_stripped and not next_stripped.startswith(("!", "#")):
                        del lines[j]
                        continue
                    break
                break
        i += 1

    if in_block and end_idx is None:
        end_idx = len(lines)

    if in_block and end_idx is not None:
        inserts = []
        for key, val in updates.items():
            if not found[key.upper()]:
                inserts.append(f"  {key} = {val}")
        if inserts:
            lines[end_idx:end_idx] = inserts

    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def _default_exec_candidates(root: Path, *, include_user_bin: bool = True) -> list[Path]:
    candidates = [
        root / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000",
        # Common local layout: STELLOPT cloned next to vmec_jax.
        root.parent / "STELLOPT" / "VMEC2000" / "Release" / "xvmec2000",
        root / "vmec2000" / "build" / "xvmec2000",
        root / "vmec2000" / "build" / "Release" / "xvmec2000",
    ]
    if include_user_bin:
        candidates.append(Path.home() / "bin" / "xvmec2000")
    return candidates


def find_vmec2000_exec(*, root: Path | None = None) -> Path | None:
    env_path = os.environ.get("VMEC2000_EXEC")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    base = root or Path(__file__).resolve().parents[2]
    for cand in _default_exec_candidates(base, include_user_bin=root is None):
        if cand.exists():
            return cand
    return None


def _infer_case_name(input_path: Path) -> str:
    name = input_path.name
    if name.startswith("input."):
        return name[len("input.") :]
    return name


def _find_threed1_file(workdir: Path, *, case: str) -> Path | None:
    direct = workdir / f"threed1.{case}"
    if direct.exists():
        return direct
    alt = workdir / f"threed1_{case}"
    if alt.exists():
        return alt
    # Fallback: any threed1* in workdir.
    matches = sorted(workdir.glob("threed1*"))
    return matches[0] if matches else None


def _relative_mgrid_file(text: str) -> str | None:
    """Return a relative VMEC ``MGRID_FILE`` reference from a namelist, if present."""
    match = _RE_MGRID_FILE.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    if not value or value.upper() in {"NONE", "DIRECT_COILS"}:
        return None
    if Path(value).is_absolute():
        return None
    return value


def run_xvmec2000(
    input_path: Path,
    *,
    exec_path: Path | None = None,
    workdir: Path | None = None,
    timeout_s: float = 60.0,
    indata_updates: dict[str, str] | None = None,
    keep_workdir: bool = False,
) -> Vmec2000ExecResult:
    """Run xvmec2000 in a temp directory and parse threed1.*."""
    exec_path = exec_path or find_vmec2000_exec()
    if exec_path is None or not exec_path.exists():
        raise FileNotFoundError("VMEC2000 executable not found. Set VMEC2000_EXEC or build STELLOPT/VMEC2000.")

    input_path = Path(input_path).resolve()
    case = _infer_case_name(input_path)

    if workdir is None:
        temp = tempfile.TemporaryDirectory(prefix="vmec2000_exec_")
        workdir_path = Path(temp.name)
    else:
        temp = None
        workdir_path = Path(workdir)
        workdir_path.mkdir(parents=True, exist_ok=True)

    try:
        dest = workdir_path / input_path.name
        shutil.copy2(input_path, dest)
        if indata_updates:
            patched = _patch_indata(dest.read_text(), updates={k.upper(): str(v) for k, v in indata_updates.items()})
            dest.write_text(patched)
        mgrid_name = _relative_mgrid_file(dest.read_text())
        if mgrid_name is not None:
            source = input_path.parent / mgrid_name
            target = workdir_path / mgrid_name
            if source.exists() and source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        cmd = [str(exec_path), dest.name]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=workdir_path,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        runtime_s = time.perf_counter() - t0

        threed1_path = _find_threed1_file(workdir_path, case=case)
        stages: list[Vmec2000Threed1Stage] = []
        if threed1_path is not None and threed1_path.exists():
            stages = _parse_vmec2000_threed1(threed1_path)

        result = Vmec2000ExecResult(
            workdir=workdir_path,
            input_path=dest,
            returncode=int(getattr(proc, "returncode", 0)),
            stdout=proc.stdout,
            stderr=proc.stderr,
            runtime_s=float(runtime_s),
            threed1_path=threed1_path,
            stages=stages,
        )
    finally:
        if (temp is not None) and (not keep_workdir):
            temp.cleanup()

    return result


def flatten_threed1(stages: Iterable[Vmec2000Threed1Stage]) -> list[Vmec2000Threed1Row]:
    rows: list[Vmec2000Threed1Row] = []
    for stage in stages:
        rows.extend(stage.rows)
    return rows


def threed1_fsq_total(rows: Iterable[Vmec2000Threed1Row]) -> np.ndarray:
    r = list(rows)
    if not r:
        return np.asarray([], dtype=float)
    return np.asarray([x.fsqr + x.fsqz + x.fsql for x in r], dtype=float)

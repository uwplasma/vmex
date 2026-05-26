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
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import math

import vmec_jax.api as vj
from vmec_jax.solve import SolveVmecResidualResult
from vmec_jax.wout import read_wout
from vmec_jax.vmec_tomnsp import vmec_trig_tables
from vmec_jax.vmec_forces import vmec_forces_rz_from_wout
from vmec_jax.wout import _compute_bsubs_half_mesh


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
    r"^\s*NS\s*=\s*(\d+)\s+NO\.\s+FOURIER\s+MODES\s*=\s*(\d+)\s+FTOLV\s*=\s*([0-9.DdEe+-]+)\s+NITER\s*=\s*([+-]?\d+)"
)
_RE_ROW = re.compile(r"^\s*(\d+)\s+([0-9.DdEe+-]+)\s+([0-9.DdEe+-]+)\s+([0-9.DdEe+-]+)\s+")
_RE_XC = re.compile(r"xc_.*_ns(\d+)_iter(\d+)\.dat$")
_RE_XC_INIT = re.compile(r"xc_init(?:_([A-Za-z0-9]+))?_ns(\d+)\.dat$")
_RE_BSUBE = re.compile(r"bsube_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_BSUBH = re.compile(r"bsubh_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_BSUBH_VMEC = re.compile(r"bsubh_iter(\d+)\.dat$")
_RE_BSUB_PRE_SYM_JAX = re.compile(r"bsub_pre_sym_jax(?:_(.*))?\.dat$")
_RE_BSUB_PRE_SYM_VMEC = re.compile(r"bsub_pre_sym(?:_(.*))?\.dat$")
_RE_BSUBE_TERMS = re.compile(r"bsube_terms_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_BSUBS = re.compile(r"bsubs_(?:ns(\d+)_)?iter(\d+)\.npz$")
_RE_BSUBS_VMEC = re.compile(r"bsubs_iter(\d+)\.dat$")
_RE_JACOBIAN_TERMS = re.compile(r"jacobian_terms_iter(\d+)\.dat$")
_RE_LULV = re.compile(r"lulv_(?:ns(\d+)_)?iter(\d+)\.(?:dat|npz)$")
_RE_GC = re.compile(r"gc_(raw|precond)_?(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_TOMNSPS = re.compile(r"tomnsps_(raw|precond)?_?(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_TOMNSPS_KERNELS = re.compile(r"tomnsps_kernels_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_FORCE_KERNELS = re.compile(r"force_kernels_(raw|precond)?_?(?:ns(\d+)_)?iter(\d+)\.npz$")
_RE_SCALARS = re.compile(r"scalars_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_GCX2 = re.compile(r"gcx2_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_FSQ1 = re.compile(r"fsq1_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_LAM = re.compile(r"lam_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_LAM_FSQ1 = re.compile(r"lam_fsql1_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_LAM_GCL = re.compile(r"lam_gcl_(?:ns(\d+)_)?iter(\d+)\.npz$")
_RE_LAMCAL_PRE = re.compile(r"lamcal_pre_(?:ns(\d+)_)?iter(\d+)\.dat$")
_RE_LAMCAL_POST = re.compile(r"lamcal_post_(?:ns(\d+)_)?iter(\d+)\.dat$")


def _parse_ns_iter(match, *, ns_group: int, iter_group: int) -> tuple[int | None, int]:
    ns_raw = match.group(ns_group) if match.group(ns_group) is not None else None
    ns_val = int(ns_raw) if ns_raw not in (None, "") else None
    it_val = int(match.group(iter_group))
    return ns_val, it_val


def _parse_int_list_arg(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    if not parts:
        return None
    return [int(float(p)) for p in parts]


def _parse_float_list_arg(value: str | None) -> list[float] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[\s,]+", text) if p]
    if not parts:
        return None
    return [float(p) for p in parts]


def _parse_fortran_float(tok: str) -> float:
    """Parse Fortran/VMEC dump float token robustly.

    Handles canonical D/E exponents and legacy tokens where `E` is omitted,
    e.g. `1.0634199273127645-314`.
    """
    t = tok.strip().replace("D", "E").replace("d", "E")
    try:
        return float(t)
    except ValueError:
        m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d{2,4})", t)
        if m:
            return float(f"{m.group(1)}E{m.group(2)}")
        return float("nan")


def _extend_list(values: list, target_len: int) -> list:
    if len(values) >= target_len:
        return values[:target_len]
    if not values:
        return values
    return values + [values[-1]] * (target_len - len(values))


def _radial_trim_axis(
    arr: np.ndarray | None,
    radial_skip: int,
    radial_drop_edge: bool,
    *,
    axis: int = 0,
) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr
    ax = axis if axis >= 0 else arr.ndim + axis
    if ax < 0 or ax >= arr.ndim:
        return arr
    n = arr.shape[ax]
    start = max(int(radial_skip), 0)
    if start >= n:
        start = n
    end = n
    if radial_drop_edge and n > start:
        end = max(start, n - 1)
    sl = [slice(None)] * arr.ndim
    sl[ax] = slice(start, end)
    return arr[tuple(sl)]


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
    has_z00 = False

    def _flush() -> None:
        nonlocal current, rows, in_table, has_z00
        if current is None:
            return
        stages.append(Vmec2000Threed1Stage(ns=current.ns, niter=current.niter, ftolv=current.ftolv, rows=rows))
        current = None
        rows = []
        in_table = False
        has_z00 = False

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
                has_z00 = False
                continue
            current = Vmec2000Threed1Stage(ns=ns, niter=niter, ftolv=ftolv, rows=[])
            continue

        if current is None:
            continue

        if line.strip().startswith("ITER") and ("FSQR" in line) and ("fsqr" in line):
            in_table = True
            has_z00 = ("ZAX" in line) or ("ZAX(v=0)" in line)
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
        if has_z00:
            r00 = _f(toks[8]) if len(toks) > 8 else None
            w_idx = 10
        else:
            r00 = _f(toks[8]) if len(toks) > 8 else None
            w_idx = 9
        r = Vmec2000Threed1Row(
            it=it,
            fsqr=_f(toks[1]),
            fsqz=_f(toks[2]),
            fsql=_f(toks[3]),
            fsqr1=_f(toks[4]),
            fsqz1=_f(toks[5]),
            fsql1=_f(toks[6]),
            delt0r=_f(toks[7]) if len(toks) > 7 else None,
            r00=r00,
            w=_f(toks[w_idx]) if len(toks) > w_idx else None,
        )
        rows.append(r)

    _flush()
    return stages


@contextmanager
def _workdir_context(path: Path | None):
    if path is None:
        with tempfile.TemporaryDirectory(prefix="vmec2000_exec_") as td:
            yield Path(td)
    else:
        path.mkdir(parents=True, exist_ok=True)
        yield path


def _copy_input_and_mgrid(input_path: Path, workdir: Path) -> Path:
    """Copy a VMEC input deck and its referenced MGRID file into ``workdir``."""

    from vmec_jax.namelist import read_indata

    repo_root = Path(__file__).resolve().parents[2]
    src = input_path.resolve()
    dst = workdir / src.name
    shutil.copy2(src, dst)
    try:
        indata = read_indata(src)
        mg = str(indata.get("MGRID_FILE", "NONE")).strip().strip("'").strip('"')
    except Exception:
        mg = "NONE"
    if not mg or mg.upper() == "NONE":
        return dst

    mg_src = Path(mg)
    if not mg_src.is_absolute():
        mg_src = src.parent / mg_src
    if not mg_src.exists():
        search_roots = [
            repo_root / "examples" / "data",
            repo_root / "examples_single_grid" / "data",
            repo_root.parent / "STELLOPT" / "BENCHMARKS" / "VMEC_TEST",
            repo_root.parent / "external",
            repo_root / "outputs",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            exact = (root / Path(mg).name).resolve()
            if exact.exists():
                mg_src = exact
                break
            matches = list(root.rglob(Path(mg).name))
            if matches:
                mg_src = matches[0].resolve()
                break
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


def _parse_vmec_xc_dump(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse VMEC2000 xc/xcdot dump (text) -> (xc, v)."""
    xc_vals: list[float] = []
    v_vals: list[float] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("neqs=") or line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        try:
            _i = int(toks[0])
        except ValueError:
            continue
        xc_vals.append(_parse_fortran_float(toks[1]))
        v_vals.append(_parse_fortran_float(toks[2]))
    return np.asarray(xc_vals, dtype=float), np.asarray(v_vals, dtype=float)


def _collect_vmec_xc_dumps(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("xc_*_iter*.dat")):
        m = _RE_XC.search(p.name)
        if not m:
            continue
        ns = int(m.group(1))
        it = int(m.group(2))
        out[(ns, it)] = _parse_vmec_xc_dump(p)
    return out


def _collect_jax_xc_dumps(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("xc_ns*_iter*.npz")):
        m = re.search(r"xc_ns(\d+)_iter(\d+)\.npz$", p.name)
        if not m:
            continue
        ns = int(m.group(1))
        it = int(m.group(2))
        data = np.load(p)
        if "v" in data:
            v = np.asarray(data["v"])
        elif "xcdot" in data:
            v = np.asarray(data["xcdot"])
        else:
            raise KeyError(f"Missing v/xcdot in {p}")
        out[(ns, it)] = (np.asarray(data["xc"]), v)
    return out


def _collect_jax_xc_init_dumps(path: Path) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Collect JAX initial-guess xc dumps keyed by (ns, label)."""
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("xc_init*_ns*.dat")):
        m = _RE_XC_INIT.match(p.name)
        if not m:
            continue
        label = (m.group(1) or "").strip()
        ns = int(m.group(2))
        out[(ns, label)] = _parse_vmec_xc_dump(p)
    return out


def _parse_vmec_gc_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse VMEC2000 gc dump (text) -> (gcr, gcz, gcl) with shape (ns, ntor+1, mpol1+1, ntmax)."""
    ns = mpol1 = ntor = ntmax = None
    rows: list[tuple[int, int, int, int, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1])
            continue
        if line.startswith("mpol1="):
            mpol1 = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntor="):
            ntor = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntmax="):
            ntmax = int(line.split("=", 1)[-1])
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 7:
            continue
        try:
            js = int(toks[0])
            m = int(toks[1])
            n = int(toks[2])
            t = int(toks[3])
        except ValueError:
            continue
        gcr = float(toks[4].replace("D", "E").replace("d", "E"))
        gcz = float(toks[5].replace("D", "E").replace("d", "E"))
        gcl = float(toks[6].replace("D", "E").replace("d", "E"))
        rows.append((js, m, n, t, gcr, gcz, gcl))

    if ns is None or mpol1 is None or ntor is None or ntmax is None:
        raise ValueError(f"Malformed gc dump: {path}")

    gcr = np.zeros((ns, ntor + 1, mpol1 + 1, ntmax), dtype=float)
    gcz = np.zeros_like(gcr)
    gcl = np.zeros_like(gcr)
    for js, m, n, t, vcr, vcz, vcl in rows:
        gcr[js - 1, n, m, t - 1] = vcr
        gcz[js - 1, n, m, t - 1] = vcz
        gcl[js - 1, n, m, t - 1] = vcl
    return gcr, gcz, gcl


def _collect_vmec_gc_dumps(path: Path) -> dict[tuple[str, int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gc_*_iter*.dat")):
        m = _RE_GC.search(p.name)
        if not m:
            continue
        stage = str(m.group(1))
        ns_val, it = _parse_ns_iter(m, ns_group=2, iter_group=3)
        out[(stage, ns_val, it)] = _parse_vmec_gc_dump(p)
    return out


def _collect_jax_gc_dumps(path: Path) -> dict[tuple[str, int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gc_*_iter*.npz")):
        m = _RE_GC.search(p.name.replace(".npz", ".dat"))
        if not m:
            continue
        stage = str(m.group(1))
        ns_val, it = _parse_ns_iter(m, ns_group=2, iter_group=3)
        data = np.load(p)
        gcr = np.asarray(data["gcr"])
        gcz = np.asarray(data["gcz"])
        gcl = np.asarray(data["gcl"])
        if gcr.ndim == 4:
            gcr = np.transpose(gcr, (0, 2, 1, 3))
            gcz = np.transpose(gcz, (0, 2, 1, 3))
            gcl = np.transpose(gcl, (0, 2, 1, 3))
        out[(stage, ns_val, it)] = (gcr, gcz, gcl)
    return out


def _parse_vmec_lam_dump(path: Path) -> dict[str, np.ndarray]:
    """Parse VMEC2000 lambda preconditioner dump (pfaclam/faclam)."""
    ns = mpol1 = ntor = ntmax = None
    rows: list[tuple[int, int, int, int, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1])
            continue
        if line.startswith("mpol1="):
            mpol1 = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntor="):
            ntor = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntmax="):
            ntmax = int(line.split("=", 1)[-1])
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            js = int(toks[0])
            m = int(toks[1])
            n = int(toks[2])
            t = int(toks[3])
        except ValueError:
            continue
        pfac = float(toks[4].replace("D", "E").replace("d", "E"))
        fac = float(toks[5].replace("D", "E").replace("d", "E")) if len(toks) > 5 else float("nan")
        rows.append((js, m, n, t, pfac, fac))

    if ns is None or mpol1 is None or ntor is None or ntmax is None:
        raise ValueError(f"Malformed lam dump: {path}")

    pfaclam = np.zeros((ns, ntor + 1, mpol1 + 1, ntmax), dtype=float)
    faclam = np.zeros_like(pfaclam)
    for js, m, n, t, pfac, fac in rows:
        pfaclam[js - 1, n, m, t - 1] = pfac
        if np.isfinite(fac):
            faclam[js - 1, n, m, t - 1] = fac
    data = {"pfaclam": pfaclam}
    if np.any(np.isfinite(faclam)):
        data["faclam"] = faclam
    return data


def _collect_vmec_lam_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lam_*_iter*.dat")):
        m = _RE_LAM.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_vmec_lam_dump(p)
    return out


def _collect_jax_lam_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lam_prec_ns*_iter*.npz")):
        m = re.search(r"lam_prec_ns(\d+)_iter(\d+)\.npz$", p.name)
        if not m:
            continue
        ns_val = int(m.group(1))
        it = int(m.group(2))
        data = np.load(p)
        pfaclam = np.asarray(data["pfaclam"])
        faclam = data["faclam"] if "faclam" in data else None
        out[(ns_val, it)] = {"pfaclam": pfaclam, "faclam": faclam}
    return out


def _parse_lam_fsql1_dump(path: Path) -> dict[str, float]:
    data: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 3:
            continue
        if not toks[0].lstrip("+-").isdigit():
            continue
        data["iter"] = float(toks[0])
        data["fsql1_pre"] = float(toks[1].replace("D", "E").replace("d", "E"))
        data["fsql1_post"] = float(toks[2].replace("D", "E").replace("d", "E"))
        break
    if not data:
        raise ValueError(f"Malformed lam_fsql1 dump: {path}")
    return data


def _collect_lam_fsql1_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, float]]:
    out: dict[tuple[int | None, int], dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lam_fsql1_*_iter*.dat")):
        m = _RE_LAM_FSQ1.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_lam_fsql1_dump(p)
    return out


def _collect_lam_gcl_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lam_gcl_*_iter*.npz")):
        m = _RE_LAM_GCL.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        data = np.load(p)
        out[(ns_val, it)] = {
            "gcl_pre": np.asarray(data["gcl_pre"]),
            "gcl_post": np.asarray(data["gcl_post"]),
            "fsql1_pre": float(np.asarray(data.get("fsql1_pre", np.nan))),
            "fsql1_post": float(np.asarray(data.get("fsql1_post", np.nan))),
        }
    return out


def _parse_vmec_tomnsps_dump(path: Path) -> dict[str, np.ndarray]:
    """Parse VMEC2000 tomnsps dump (text) -> dict of blocks (ns, mpol1+1, ntor+1)."""
    ns = mpol1 = ntor = None
    rows: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1])
            continue
        if line.startswith("mpol1="):
            mpol1 = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntor="):
            ntor = int(line.split("=", 1)[-1])
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 9:
            continue
        try:
            js = int(toks[0])
            m = int(toks[1])
            n = int(toks[2])
        except ValueError:
            continue
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[3:9]]
        rows.append((js, m, n, *vals))

    if ns is None or mpol1 is None or ntor is None:
        raise ValueError(f"Malformed tomnsps dump: {path}")

    shape = (ns, mpol1 + 1, ntor + 1)
    frcc = np.zeros(shape, dtype=float)
    frss = np.zeros(shape, dtype=float)
    fzsc = np.zeros(shape, dtype=float)
    fzcs = np.zeros(shape, dtype=float)
    flsc = np.zeros(shape, dtype=float)
    flcs = np.zeros(shape, dtype=float)
    for js, m, n, v_frcc, v_frss, v_fzsc, v_fzcs, v_flsc, v_flcs in rows:
        frcc[js - 1, m, n] = v_frcc
        frss[js - 1, m, n] = v_frss
        fzsc[js - 1, m, n] = v_fzsc
        fzcs[js - 1, m, n] = v_fzcs
        flsc[js - 1, m, n] = v_flsc
        flcs[js - 1, m, n] = v_flcs
    return {
        "frcc": frcc,
        "frss": frss,
        "fzsc": fzsc,
        "fzcs": fzcs,
        "flsc": flsc,
        "flcs": flcs,
    }


def _parse_vmec_tomnsps_kernels_dump(path: Path) -> dict[str, np.ndarray]:
    """Parse VMEC2000 tomnsps kernels dump -> dict of (ns, ntheta3, nzeta, 2) arrays."""
    ns = None
    ntheta3 = None
    nzeta = None
    rows: list[tuple[int, int, int, int, list[float]]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1])
            continue
        if line.startswith("ntheta3="):
            ntheta3 = int(line.split("=", 1)[-1])
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1])
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 14:
            continue
        try:
            js = int(toks[0])
            lt = int(toks[1])
            lz = int(toks[2])
            mpar = int(toks[3])
        except ValueError:
            continue
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[4:14]]
        rows.append((js, lt, lz, mpar, vals))

    if ns is None or ntheta3 is None or nzeta is None:
        raise ValueError(f"Malformed tomnsps_kernels dump: {path}")

    shape = (ns, ntheta3, nzeta, 2)
    names = ("armn", "brmn", "crmn", "azmn", "bzmn", "czmn", "arcon", "azcon", "blmn", "clmn")
    out = {name: np.zeros(shape, dtype=float) for name in names}
    for js, lt, lz, mpar, vals in rows:
        if not (0 <= mpar <= 1):
            continue
        idx = (js - 1, lt - 1, lz - 1, mpar)
        for name, v in zip(names, vals, strict=True):
            out[name][idx] = v
    return out


def _collect_vmec_tomnsps_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps*_iter*.dat")):
        m = _RE_TOMNSPS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=2, iter_group=3)
        out[(ns_val, it)] = _parse_vmec_tomnsps_dump(p)
    return out


def _collect_vmec_tomnsps_kernels_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps_kernels*_iter*.dat")):
        m = _RE_TOMNSPS_KERNELS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_vmec_tomnsps_kernels_dump(p)
    return out


def _collect_jax_tomnsps_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps_raw*_iter*.npz")):
        m = _RE_TOMNSPS.search(p.name.replace(".npz", ".dat"))
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=2, iter_group=3)
        data = np.load(p)
        frcc = np.asarray(data["frcc"])
        shape = frcc.shape
        def _block(name: str) -> np.ndarray:
            arr = np.asarray(data[name])
            if arr.size == 0:
                return np.zeros(shape, dtype=frcc.dtype)
            return arr
        out[(ns_val, it)] = {
            "frcc": frcc,
            "frss": _block("frss"),
            "fzsc": _block("fzsc"),
            "fzcs": _block("fzcs"),
            "flsc": _block("flsc"),
            "flcs": _block("flcs"),
        }
    return out


def _collect_jax_force_kernels(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("force_kernels_raw*_iter*.npz")):
        m = _RE_FORCE_KERNELS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=2, iter_group=3)
        data = np.load(p)
        ns = int(np.asarray(data.get("ns", 0)))
        nzeta = int(np.asarray(data.get("nzeta", 0)))
        default_shape = None
        for key in ("armn_e", "brmn_e", "azmn_e", "blmn_e", "clmn_e"):
            if key in data and np.asarray(data[key]).size > 0:
                default_shape = tuple(np.asarray(data[key]).shape)
                break
        if default_shape is None:
            ntheta = int(np.asarray(data.get("ntheta", 0)))
            default_shape = (ns, ntheta, nzeta)

        def _get(name: str) -> np.ndarray:
            if name not in data:
                return np.zeros(default_shape, dtype=float)
            arr = np.asarray(data[name])
            if arr.size == 0:
                return np.zeros(default_shape, dtype=float)
            if arr.shape != default_shape:
                if arr.size == int(np.prod(default_shape)):
                    return arr.reshape(default_shape)
            return arr

        def _parity(even_name: str, odd_name: str) -> np.ndarray:
            even = _get(even_name)
            odd = _get(odd_name)
            return np.stack([even, odd], axis=-1)

        out[(ns_val, it)] = {
            "armn": _parity("armn_e", "armn_o"),
            "brmn": _parity("brmn_e", "brmn_o"),
            "crmn": _parity("crmn_e", "crmn_o"),
            "azmn": _parity("azmn_e", "azmn_o"),
            "bzmn": _parity("bzmn_e", "bzmn_o"),
            "czmn": _parity("czmn_e", "czmn_o"),
            "arcon": _parity("arcon_e", "arcon_o"),
            "azcon": _parity("azcon_e", "azcon_o"),
            "blmn": _parity("blmn_e", "blmn_o"),
            "clmn": _parity("clmn_e", "clmn_o"),
            "ru12": _get("ru12"),
            "zs": _get("zs"),
            "zu12": _get("zu12"),
            "rs": _get("rs"),
            "r12": _get("r12"),
            "tau": _get("tau"),
            "sqrtg": _get("sqrtg"),
            "pru_even": _get("pru_even"),
            "pru_odd": _get("pru_odd"),
            "pz1_even": _get("pz1_even"),
            "pz1_odd": _get("pz1_odd"),
            "pzu_even": _get("pzu_even"),
            "pzu_odd": _get("pzu_odd"),
            "pr1_even": _get("pr1_even"),
            "pr1_odd": _get("pr1_odd"),
            "prv_even": _get("prv_even"),
            "prv_odd": _get("prv_odd"),
            "pzv_even": _get("pzv_even"),
            "pzv_odd": _get("pzv_odd"),
        }
    return out


def _parse_scalars_dump(path: Path) -> dict[str, float]:
    vals: list[float] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("cols:"):
            continue
        toks = line.split()
        if not toks:
            continue
        if toks[0].lstrip("+-").isdigit():
            for t in toks[1:]:
                vals.append(float(t.replace("D", "E").replace("d", "E")))
            break
    if len(vals) < 6:
        raise ValueError(f"Malformed scalars dump: {path}")
    # Order: wb wp volume r2 fnorm fnorm1 fnormL
    out = {
        "wb": float(vals[0]),
        "wp": float(vals[1]),
        "volume": float(vals[2]),
        "r2": float(vals[3]),
        "fnorm": float(vals[4]),
        "fnorm1": float(vals[5]) if len(vals) > 5 else float("nan"),
        "fnormL": float(vals[6]) if len(vals) > 6 else float("nan"),
    }
    return out


def _collect_scalars_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, float]]:
    out: dict[tuple[int | None, int], dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("scalars*_iter*.dat")):
        m = _RE_SCALARS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_scalars_dump(p)
    return out


def _parse_gcx2_dump(path: Path) -> dict[str, float]:
    data: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        if not toks[0].lstrip("+-").isdigit():
            continue
        data["iter"] = float(toks[0])
        data["include_edge"] = float(toks[1])
        data["gcr2"] = float(toks[2].replace("D", "E").replace("d", "E"))
        data["gcz2"] = float(toks[3].replace("D", "E").replace("d", "E"))
        data["gcl2"] = float(toks[4].replace("D", "E").replace("d", "E"))
        break
    if not data:
        raise ValueError(f"Malformed gcx2 dump: {path}")
    return data


def _collect_gcx2_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, float]]:
    out: dict[tuple[int | None, int], dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gcx2*_iter*.dat")):
        m = _RE_GCX2.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_gcx2_dump(p)
    return out


def _parse_fsq1_dump(path: Path) -> dict[str, float]:
    data: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        if not toks[0].lstrip("+-").isdigit():
            continue
        data["iter"] = float(toks[0])
        data["fsqr1"] = float(toks[1].replace("D", "E").replace("d", "E"))
        data["fsqz1"] = float(toks[2].replace("D", "E").replace("d", "E"))
        data["fsql1"] = float(toks[3].replace("D", "E").replace("d", "E"))
        break
    if not data:
        raise ValueError(f"Malformed fsq1 dump: {path}")
    return data


def _parse_time_control_trace(path: Path) -> dict[int, float]:
    """Parse time_control_trace.log and return iter2 -> delt0r (pre stage)."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        toks = line.split()
        if len(toks) < 9:
            continue
        try:
            iter2 = int(toks[0])
            stage = toks[8]
            if stage != "pre":
                continue
            delt0r = float(toks[6])
        except Exception:
            continue
        if iter2 not in out:
            out[iter2] = delt0r
    return out


def _collect_fsq1_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, float]]:
    out: dict[tuple[int | None, int], dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("fsq1*_iter*.dat")):
        m = _RE_FSQ1.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_fsq1_dump(p)
    return out


def _offset_dump_keys(dumps: dict, *, offset: int) -> dict:
    """Offset iteration index in dump dict keys by `offset`."""
    if offset == 0:
        return dumps
    out: dict = {}
    for key, val in dumps.items():
        if not isinstance(key, tuple) or len(key) == 0:
            out[key] = val
            continue
        *prefix, it = key
        out[(*prefix, int(it) + int(offset))] = val
    return out


def _merge_dump_dicts(d1: dict, d2: dict, *, offset: int, overwrite: bool = True) -> dict:
    out = dict(d1)
    d2_shifted = _offset_dump_keys(d2, offset=offset)
    if overwrite:
        out.update(d2_shifted)
    else:
        for key, value in d2_shifted.items():
            if key not in out:
                out[key] = value
    return out


def _merge_vmec_results(res_a: SolveVmecResidualResult | None, res_b: SolveVmecResidualResult | None) -> SolveVmecResidualResult | None:
    if res_a is None:
        return res_b
    if res_b is None:
        return res_a

    def _cat(a, b):
        if a is None and b is None:
            return np.zeros((0,), dtype=float)
        a_arr = np.asarray(a) if a is not None else np.zeros((0,), dtype=float)
        b_arr = np.asarray(b) if b is not None else np.zeros((0,), dtype=float)
        if a_arr.ndim == 0:
            a_arr = a_arr.reshape((1,))
        if b_arr.ndim == 0:
            b_arr = b_arr.reshape((1,))
        return np.concatenate([a_arr, b_arr], axis=0)

    diag = {}
    keys = set(res_a.diagnostics.keys()) | set(res_b.diagnostics.keys())
    for k in keys:
        v1 = res_a.diagnostics.get(k)
        v2 = res_b.diagnostics.get(k)
        if k.startswith("multigrid_"):
            diag[k] = v2 if v2 is not None else v1
            continue
        if isinstance(v1, (list, tuple, np.ndarray)) or isinstance(v2, (list, tuple, np.ndarray)):
            diag[k] = _cat(v1, v2)
        else:
            diag[k] = v2 if v2 is not None else v1

    w_history = _cat(res_a.w_history, res_b.w_history)
    return SolveVmecResidualResult(
        state=res_b.state,
        n_iter=int(len(w_history) - 1),
        w_history=w_history,
        fsqr2_history=_cat(res_a.fsqr2_history, res_b.fsqr2_history),
        fsqz2_history=_cat(res_a.fsqz2_history, res_b.fsqz2_history),
        fsql2_history=_cat(res_a.fsql2_history, res_b.fsql2_history),
        grad_rms_history=_cat(res_a.grad_rms_history, res_b.grad_rms_history),
        step_history=_cat(res_a.step_history, res_b.step_history),
        diagnostics=diag,
    )


def _compute_fsq_from_dumps(
    *,
    scalars: dict[tuple[int | None, int], dict[str, float]],
    gcx2: dict[tuple[int | None, int], dict[str, float]],
    r1: float,
) -> dict[tuple[int | None, int], dict[str, float]]:
    out: dict[tuple[int | None, int], dict[str, float]] = {}
    if not scalars or not gcx2:
        return out
    scalar_keys = sorted(scalars.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))

    def _lookup_scalar(ns_val: int | None, it: int) -> dict[str, float] | None:
        if (ns_val, it) in scalars:
            return scalars[(ns_val, it)]
        if (None, it) in scalars:
            return scalars[(None, it)]
        best_it = -1
        best = None
        for ns_k, it_k in scalar_keys:
            if ns_k is not None and ns_val is not None and ns_k != ns_val:
                continue
            if it_k <= it and it_k > best_it:
                best_it = it_k
                best = scalars[(ns_k, it_k)]
        return best

    for key in sorted(gcx2.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1])):
        ns_val, it = key
        sc = _lookup_scalar(ns_val, it)
        if sc is None:
            continue
        gc = gcx2[key]
        fnorm = float(sc.get("fnorm", float("nan")))
        fnormL = float(sc.get("fnormL", float("nan")))
        gcr2 = float(gc.get("gcr2", float("nan")))
        gcz2 = float(gc.get("gcz2", float("nan")))
        gcl2 = float(gc.get("gcl2", float("nan")))
        out[(ns_val, it)] = {
            "fsqr": r1 * fnorm * gcr2,
            "fsqz": r1 * fnorm * gcz2,
            "fsql": fnormL * gcl2,
        }
    return out


def _parse_bsube_dump(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ns = None
    ntheta = None
    nzeta = None
    data: list[tuple[int, int, int, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        bsubu = float(toks[3].replace("D", "E").replace("d", "E"))
        bsubv = float(toks[4].replace("D", "E").replace("d", "E"))
        data.append((js, lt, lz, bsubu, bsubv))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    bsubu_arr = np.zeros((ns, ntheta, nzeta), dtype=float)
    bsubv_arr = np.zeros_like(bsubu_arr)
    for js, lt, lz, bsubu, bsubv in data:
        bsubu_arr[js, lt, lz] = bsubu
        bsubv_arr[js, lt, lz] = bsubv
    return bsubu_arr, bsubv_arr


def _parse_bsube_terms_dump(path: Path) -> tuple[dict[str, np.ndarray], int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 9:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[3:9]]
        rows.append((js, lt, lz, *vals))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    lvv_sh = np.zeros((ns, ntheta, nzeta), dtype=float)
    lu0 = np.zeros_like(lvv_sh)
    lu1 = np.zeros_like(lvv_sh)
    phipf = np.zeros_like(lvv_sh)
    bsubu_tmp = np.zeros_like(lvv_sh)
    bsubv_pre = np.zeros_like(lvv_sh)
    for js, lt, lz, v_lvv, v_lu0, v_lu1, v_phip, v_bu, v_bv in rows:
        lvv_sh[js, lt, lz] = v_lvv
        lu0[js, lt, lz] = v_lu0
        lu1[js, lt, lz] = v_lu1
        phipf[js, lt, lz] = v_phip
        bsubu_tmp[js, lt, lz] = v_bu
        bsubv_pre[js, lt, lz] = v_bv
    return (
        {
            "lvv_sh": lvv_sh,
            "lu0": lu0,
            "lu1": lu1,
            "phipf": phipf,
            "bsubu_tmp": bsubu_tmp,
            "bsubv_pre": bsubv_pre,
        },
        int(ns),
    )


def _parse_bsubs_dump(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    out = {
        "bsubs_half": np.asarray(data["bsubs_half"]),
        "bsubs_full": np.asarray(data["bsubs_full"]),
        "bsupu": np.asarray(data["bsupu"]),
        "bsupv": np.asarray(data["bsupv"]),
        "bsupu1": np.asarray(data["bsupu1"]),
        "bsupv1": np.asarray(data["bsupv1"]),
        "bsubu": np.asarray(data["bsubu"]),
        "bsubv": np.asarray(data["bsubv"]),
        "sqrtg": np.asarray(data["sqrtg"]),
    }
    if "bsubu_jxbout" in data:
        out["bsubu_jxbout"] = np.asarray(data["bsubu_jxbout"])
    if "bsubv_jxbout" in data:
        out["bsubv_jxbout"] = np.asarray(data["bsubv_jxbout"])
    if "bsubu_raw" in data:
        out["bsubu_raw"] = np.asarray(data["bsubu_raw"])
    if "bsubv_raw" in data:
        out["bsubv_raw"] = np.asarray(data["bsubv_raw"])
    if "s" in data:
        out["s"] = np.asarray(data["s"])
    return out


def _parse_bsubh_dump(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        val_u = float(toks[3].replace("D", "E").replace("d", "E"))
        val_v = float(toks[4].replace("D", "E").replace("d", "E"))
        rows.append((js, lt, lz, val_u, val_v))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    bsubu = np.zeros((ns, ntheta, nzeta), dtype=float)
    bsubv = np.zeros_like(bsubu)
    for js, lt, lz, val_u, val_v in rows:
        bsubu[js, lt, lz] = val_u
        bsubv[js, lt, lz] = val_v
    return bsubu, bsubv, int(ns)


def _parse_bsub_pre_sym_dump(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 8:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(t.replace("D", "E").replace("d", "E")) for t in toks[3:8]]
        rows.append((js, lt, lz, *vals))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    bsubu = np.zeros((ns, ntheta, nzeta), dtype=float)
    bsubv = np.zeros_like(bsubu)
    bsupu = np.zeros_like(bsubu)
    bsupv = np.zeros_like(bsubu)
    bsubs = np.zeros_like(bsubu)
    for js, lt, lz, val_u, val_v, val_su, val_sv, val_s in rows:
        bsubu[js, lt, lz] = val_u
        bsubv[js, lt, lz] = val_v
        bsupu[js, lt, lz] = val_su
        bsupv[js, lt, lz] = val_sv
        bsubs[js, lt, lz] = val_s
    return bsubu, bsubv, bsupu, bsupv, bsubs


def _parse_vmec_bsubs_dump(path: Path) -> tuple[np.ndarray, int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        val = float(toks[3].replace("D", "E").replace("d", "E"))
        rows.append((js, lt, lz, val))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    bsubs = np.zeros((ns, ntheta, nzeta), dtype=float)
    for js, lt, lz, val in rows:
        bsubs[js, lt, lz] = val
    return bsubs, int(ns)


def _parse_vmec_bss_inputs(path: Path) -> tuple[dict[str, np.ndarray], int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 10:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[3:10]]
        if len(vals) != 7:
            continue
        r12, bzmn, brmn, azmn, armn, bsupu, bsupv = vals
        rows.append((js, lt, lz, r12, bzmn, brmn, azmn, armn, bsupu, bsupv))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    shape = (ns, ntheta, nzeta)
    out = {
        "r12": np.zeros(shape, dtype=float),
        "rs": np.zeros(shape, dtype=float),
        "zs": np.zeros(shape, dtype=float),
        "ru12": np.zeros(shape, dtype=float),
        "zu12": np.zeros(shape, dtype=float),
        "bsupu": np.zeros(shape, dtype=float),
        "bsupv": np.zeros(shape, dtype=float),
    }
    for js, lt, lz, r12, bzmn, brmn, azmn, armn, bsupu, bsupv in rows:
        out["r12"][js, lt, lz] = r12
        out["rs"][js, lt, lz] = bzmn
        out["zs"][js, lt, lz] = brmn
        out["ru12"][js, lt, lz] = azmn
        out["zu12"][js, lt, lz] = armn
        out["bsupu"][js, lt, lz] = bsupu
        out["bsupv"][js, lt, lz] = bsupv
    return out, int(ns)


def _parse_jax_bss_inputs(path: Path) -> tuple[dict[str, np.ndarray], int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float, float, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 10:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[3:10]]
        if len(vals) != 7:
            continue
        r12, rs, zs, ru12, zu12, bsupu, bsupv = vals
        rows.append((js, lt, lz, r12, rs, zs, ru12, zu12, bsupu, bsupv))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    shape = (ns, ntheta, nzeta)
    out = {
        "r12": np.zeros(shape, dtype=float),
        "rs": np.zeros(shape, dtype=float),
        "zs": np.zeros(shape, dtype=float),
        "ru12": np.zeros(shape, dtype=float),
        "zu12": np.zeros(shape, dtype=float),
        "bsupu": np.zeros(shape, dtype=float),
        "bsupv": np.zeros(shape, dtype=float),
    }
    for js, lt, lz, r12, rs, zs, ru12, zu12, bsupu, bsupv in rows:
        out["r12"][js, lt, lz] = r12
        out["rs"][js, lt, lz] = rs
        out["zs"][js, lt, lz] = zs
        out["ru12"][js, lt, lz] = ru12
        out["zu12"][js, lt, lz] = zu12
        out["bsupu"][js, lt, lz] = bsupu
        out["bsupv"][js, lt, lz] = bsupv
    return out, int(ns)


def _parse_vmec_bss_terms(path: Path) -> tuple[dict[str, np.ndarray], int]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, list[float]]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 3 + 23:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[3:]]
        if len(vals) < 23:
            continue
        rows.append((js, lt, lz, vals))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    shape = (ns, ntheta, nzeta)
    out = {
        "r1_even": np.zeros(shape, dtype=float),
        "r1_odd": np.zeros(shape, dtype=float),
        "pr1_even": np.zeros(shape, dtype=float),
        "pr1_odd": np.zeros(shape, dtype=float),
        "rv_even": np.zeros(shape, dtype=float),
        "rv_odd": np.zeros(shape, dtype=float),
        "prv_even": np.zeros(shape, dtype=float),
        "prv_odd": np.zeros(shape, dtype=float),
        "ru_even": np.zeros(shape, dtype=float),
        "ru_odd": np.zeros(shape, dtype=float),
        "zu_even": np.zeros(shape, dtype=float),
        "zu_odd": np.zeros(shape, dtype=float),
    }
    for js, lt, lz, vals in rows:
        out["r1_even"][js, lt, lz] = vals[0]
        out["r1_odd"][js, lt, lz] = vals[1]
        out["pr1_even"][js, lt, lz] = vals[2]
        out["pr1_odd"][js, lt, lz] = vals[3]
        out["rv_even"][js, lt, lz] = vals[4]
        out["rv_odd"][js, lt, lz] = vals[5]
        out["prv_even"][js, lt, lz] = vals[6]
        out["prv_odd"][js, lt, lz] = vals[7]
        out["ru_even"][js, lt, lz] = vals[19]
        out["ru_odd"][js, lt, lz] = vals[20]
        out["zu_even"][js, lt, lz] = vals[21]
        out["zu_odd"][js, lt, lz] = vals[22]
    return out, int(ns)


def _parse_jax_bss_terms(path: Path) -> tuple[dict[str, np.ndarray], int | None]:
    data = np.load(path)
    out = {
        "r1_even": np.asarray(data.get("r1_even")),
        "r1_odd": np.asarray(data.get("r1_odd")),
        "pr1_even": np.asarray(data.get("pr1_even")),
        "pr1_odd": np.asarray(data.get("pr1_odd")),
        "rv_even": np.asarray(data.get("rv_even")),
        "rv_odd": np.asarray(data.get("rv_odd")),
        "prv_even": np.asarray(data.get("prv_even")),
        "prv_odd": np.asarray(data.get("prv_odd")),
        "ru_even": np.asarray(data.get("pru_even")),
        "ru_odd": np.asarray(data.get("pru_odd")),
        "zu_even": np.asarray(data.get("pzu_even")),
        "zu_odd": np.asarray(data.get("pzu_odd")),
    }
    ns = None
    try:
        ns = int(out["r1_even"].shape[0])
    except Exception:
        ns = None
    return out, ns


def _read_jxbout(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        import netCDF4 as nc  # type: ignore
    except Exception:
        try:
            from scipy.io import netcdf  # type: ignore
        except Exception:
            return None
        with netcdf.netcdf_file(path, "r", mmap=False) as ds:
            out = {}
            for name in ("bsupu", "bsupv", "bsubu", "bsubv", "bsubs", "sqrt_g__bdotk"):
                if name in ds.variables:
                    out[name] = np.asarray(ds.variables[name][:])
            return out
    with nc.Dataset(path) as ds:
        out = {}
        for name in ("bsupu", "bsupv", "bsubu", "bsubv", "bsubs", "sqrt_g__bdotk"):
            if name in ds.variables:
                out[name] = np.asarray(ds.variables[name][:])
        return out


def _parse_lulv_vmec_dump(path: Path) -> dict[str, np.ndarray]:
    ns = ntheta = nzeta = None
    rows: list[tuple[int, int, int, float, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 7:
            continue
        try:
            js = int(toks[0]) - 1
            lt = int(toks[1]) - 1
            lz = int(toks[2]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[3:7]]
        rows.append((js, lt, lz, *vals))
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    lu0 = np.zeros((ns, ntheta, nzeta), dtype=float)
    lu1 = np.zeros_like(lu0)
    lv0 = np.zeros_like(lu0)
    lv1 = np.zeros_like(lu0)
    for js, lt, lz, v_lu0, v_lu1, v_lv0, v_lv1 in rows:
        lu0[js, lt, lz] = v_lu0
        lu1[js, lt, lz] = v_lu1
        lv0[js, lt, lz] = v_lv0
        lv1[js, lt, lz] = v_lv1
    return {"lu0": lu0, "lu1": lu1, "lv0": lv0, "lv1": lv1}


def _parse_lulv_jax_dump(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "lu0": np.asarray(data["lu0_full"]),
        "lu1": np.asarray(data["lu1_full"]),
        "lv0": np.asarray(data["lv0_full"]),
        "lv1": np.asarray(data["lv1_full"]),
    }


def _parse_lamcal_vmec_dump(path: Path) -> dict[str, np.ndarray]:
    ns = None
    rows: list[tuple[int, float, float, float]] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("columns:"):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            js = int(toks[0]) - 1
        except ValueError:
            continue
        vals = [float(tok.replace("D", "E").replace("d", "E")) for tok in toks[1:4]]
        rows.append((js, *vals))
    if ns is None:
        raise ValueError(f"Missing header fields in {path}")
    blam = np.zeros((ns,), dtype=float)
    clam = np.zeros_like(blam)
    dlam = np.zeros_like(blam)
    for js, v_b, v_c, v_d in rows:
        blam[js] = v_b
        clam[js] = v_c
        dlam[js] = v_d
    return {"blam": blam, "clam": clam, "dlam": dlam}


def _parse_lamcal_jax_dump(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "blam_pre": np.asarray(data["blam_pre"]),
        "clam_pre": np.asarray(data["clam_pre"]),
        "dlam_pre": np.asarray(data["dlam_pre"]),
        "blam_post": np.asarray(data["blam_post"]),
        "clam_post": np.asarray(data["clam_post"]),
        "dlam_post": np.asarray(data["dlam_post"]),
    }


def _collect_bsube_dumps(path: Path) -> dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsube*_iter*.dat")):
        m = _RE_BSUBE.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_bsube_dump(p)
    return out


def _collect_bsubh_dumps(path: Path) -> dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsubh*_iter*.dat")):
        m = _RE_BSUBH.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        bsubu, bsubv, ns_val2 = _parse_bsubh_dump(p)
        if ns_val is None:
            ns_val = ns_val2
        out[(ns_val, it)] = (bsubu, bsubv)
    return out


def _collect_vmec_bsubh_dumps(path: Path) -> dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsubh_iter*.dat")):
        m = _RE_BSUBH_VMEC.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        bsubu, bsubv, ns_val = _parse_bsubh_dump(p)
        out[(ns_val, it)] = (bsubu, bsubv)
    return out


def _collect_bsub_pre_sym_dumps(
    path: Path,
) -> dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsub_pre_sym_jax*.dat")):
        if not _RE_BSUB_PRE_SYM_JAX.search(p.name):
            continue
        out[(None, 0)] = _parse_bsub_pre_sym_dump(p)
    return out


def _collect_vmec_bsub_pre_sym_dumps(
    path: Path,
) -> dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[int | None, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsub_pre_sym*.dat")):
        if not _RE_BSUB_PRE_SYM_VMEC.search(p.name):
            continue
        out[(None, 0)] = _parse_bsub_pre_sym_dump(p)
    return out


def _collect_bsubs_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsubs*_iter*.npz")):
        m = _RE_BSUBS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_bsubs_dump(p)
    return out


def _collect_vmec_bsubs_dumps(path: Path) -> dict[tuple[int | None, int], np.ndarray]:
    out: dict[tuple[int | None, int], np.ndarray] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsubs_iter*.dat")):
        m = _RE_BSUBS_VMEC.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        bsubs, ns_val = _parse_vmec_bsubs_dump(p)
        out[(ns_val, it)] = bsubs
    return out


def _collect_vmec_bss_inputs(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bss_inputs_iter*.dat")):
        m = re.match(r"bss_inputs_iter(\\d+)\\.dat$", p.name)
        if not m:
            continue
        it = int(m.group(1))
        data, _ns_val = _parse_vmec_bss_inputs(p)
        out[it] = data
    return out


def _collect_vmec_bss_terms(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bss_terms_iter*.dat")):
        m = re.match(r"bss_terms_iter(\\d+)\\.dat$", p.name)
        if not m:
            continue
        it = int(m.group(1))
        data, _ns_val = _parse_vmec_bss_terms(p)
        out[it] = data
    return out


def _collect_jax_bss_inputs(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    cands = sorted(path.glob("bss_inputs_jax*.dat"))
    if not cands:
        return None
    data, _ns_val = _parse_jax_bss_inputs(cands[-1])
    return data


def _collect_jax_bss_terms(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    cands = sorted(path.glob("bss_terms_jax*.npz"))
    if not cands:
        return None
    data, _ns_val = _parse_jax_bss_terms(cands[-1])
    return data


def _dump_bsubs_final(run: vj.FixedBoundaryRun, outdir: Path) -> None:
    """Compute and dump bsubs from the final vmec_jax state for jxbout parity."""
    if run is None or run.state is None or run.static is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    state = run.state
    static = run.static
    indata = run.indata
    ns = int(static.cfg.ns)
    iter_idx = int(run.result.n_iter) if run.result is not None else 0

    class _WoutLike:
        __slots__ = (
            "nfp",
            "mpol",
            "ntor",
            "lasym",
            "signgs",
            "icurv",
            "ncurr",
            "lcurrent",
            "flux_is_internal",
        )

        def __init__(
            self,
            *,
            nfp: int,
            mpol: int,
            ntor: int,
            lasym: bool,
            signgs: int,
            icurv,
            ncurr: int,
            lcurrent: bool,
            flux_is_internal: bool,
        ):
            self.nfp = int(nfp)
            self.mpol = int(mpol)
            self.ntor = int(ntor)
            self.lasym = bool(lasym)
            self.signgs = int(signgs)
            self.icurv = icurv
            self.ncurr = int(ncurr)
            self.lcurrent = bool(lcurrent)
            self.flux_is_internal = bool(flux_is_internal)

    from vmec_jax.solve import _icurv_full_mesh_from_indata

    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=np.asarray(static.s), signgs=int(run.signgs))
    ncurr = int(indata.get_int("NCURR", 0))
    lcurrent = bool(indata.get_bool("LCURRENT", True))

    wout_like = _WoutLike(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=int(run.signgs),
        icurv=icurv,
        ncurr=ncurr,
        lcurrent=lcurrent,
        flux_is_internal=True,
    )

    from vmec_jax.modes import nyquist_mode_table_from_grid

    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    mmax_trig = int(np.max(np.asarray(nyq_modes.m))) if nyq_modes.m.size else max(int(static.cfg.mpol) - 1, 0)
    nmax_trig = int(np.max(np.abs(np.asarray(nyq_modes.n)))) if nyq_modes.n.size else max(int(static.cfg.ntor), 0)

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(static.cfg.nfp),
        mmax=mmax_trig,
        nmax=nmax_trig,
        lasym=bool(static.cfg.lasym),
        dtype=np.asarray(state.Rcos).dtype,
    )

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=indata,
        use_wout_bsup=False,
        use_vmec_synthesis=True,
        trig=trig,
    )
    bc = k.bc
    s = np.asarray(static.s, dtype=float)
    bsupu = np.asarray(bc.bsupu)
    bsupv = np.asarray(bc.bsupv)
    bsubu = np.asarray(bc.bsubu)
    bsubv = np.asarray(bc.bsubv)
    bsubu_raw = bsubu.copy()
    bsubv_raw = bsubv.copy()
    sqrtg = np.asarray(bc.jac.sqrtg)

    bsubs_half = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=static.modes,
        s=s,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        lthreed=bool(static.cfg.ntor > 0),
        lasym=bool(static.cfg.lasym),
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom={
            "pr1_odd": np.asarray(k.pr1_odd),
            "pz1_odd": np.asarray(k.pz1_odd),
            "prv_even": np.asarray(k.prv_even),
            "prv_odd": np.asarray(k.prv_odd),
            "pzv_even": np.asarray(k.pzv_even),
            "pzv_odd": np.asarray(k.pzv_odd),
        },
        jac_half=bc.jac,
    )
    bsubs_full = np.asarray(bsubs_half, dtype=float).copy()
    if ns > 2:
        bsubs_full[1:-1] = 0.5 * (bsubs_full[1:-1] + bsubs_full[2:])
    if ns > 0:
        bsubs_full[0] = 0.0
        bsubs_full[-1] = 0.0

    bsupu1 = np.zeros_like(bsupu)
    bsupv1 = np.zeros_like(bsupv)
    if ns > 1:
        sqrtg_half = 0.5 * (sqrtg[1:] + sqrtg[:-1])
        denom = np.where(sqrtg_half != 0.0, sqrtg_half, 1.0)
        if ns > 2:
            # VMEC jxbforce averages js with js+1.
            bsupu1[1:-1] = 0.5 * (bsupu[1:-1] * sqrtg[1:-1] + bsupu[2:] * sqrtg[2:]) / denom[1:]
            bsupv1[1:-1] = 0.5 * (bsupv[1:-1] * sqrtg[1:-1] + bsupv[2:] * sqrtg[2:]) / denom[1:]
        bsupu1[0] = 0.0
        bsupu1[-1] = 0.0
        bsupv1[0] = 0.0
        bsupv1[-1] = 0.0

    # Apply VMEC jxbforce-style bsubv equilibration and low-pass filter so
    # the diagnostic dump can be compared directly to jxbout outputs.
    from vmec_jax.fourier import build_helical_basis
    from vmec_jax.vmec_tomnsp import vmec_angle_grid
    from vmec_jax.wout import (
        _apply_bsubv_equif_correction,
        _bsubuv_parity_from_coeffs,
        _filter_bsubuv_jxbforce_parity_loop,
        _pshalf_from_s,
        _vmec_wrout_nyquist_cos_coeffs,
    )

    iequi = int(indata.get_int("IEQUI", 0))
    if iequi == 1 and getattr(bc, "bsubv_preblend", None) is not None:
        bsubv = _apply_bsubv_equif_correction(
            bsubv=bsubv,
            bsubv_e=np.asarray(bc.bsubv_preblend),
            trig=trig,
        )
    if not bool(static.cfg.lasym):
        # Reuse the Nyquist mode table to match VMEC jxbforce output.
        grid_nyq = vmec_angle_grid(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            lasym=bool(static.cfg.lasym),
        )
        basis_nyq = build_helical_basis(nyq_modes, grid_nyq, cache=True)
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs(
            f=bsubu, modes=nyq_modes, trig=trig
        )
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs(
            f=bsubv, modes=nyq_modes, trig=trig
        )
        z2 = np.zeros_like(bsubumnc)
        bsubu_even, bsubu_odd, bsubv_even, bsubv_odd = _bsubuv_parity_from_coeffs(
            bsubumnc=bsubumnc,
            bsubumns=z2,
            bsubvmnc=bsubvmnc,
            bsubvmns=z2,
            modes=nyq_modes,
            basis=basis_nyq,
        )
        # Convert internal odd component to the stored (physical) odd component
        # expected by jxbforce before the parity filter.
        pshalf = _pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
        if pshalf.shape[0] > 1:
            pshalf[0] = pshalf[1]
        bsubu_odd = bsubu_odd * pshalf
        bsubv_odd = bsubv_odd * pshalf
        bsubu, bsubv = _filter_bsubuv_jxbforce_parity_loop(
            bsubu_even=bsubu_even,
            bsubu_odd=bsubu_odd,
            bsubv_even=bsubv_even,
            bsubv_odd=bsubv_odd,
            trig=trig,
            mmax_force=max(int(static.cfg.mpol) - 1, 0),
            nmax_force=int(static.cfg.ntor),
            s=s,
        )

    bsubu_jxbout = np.asarray(bsubu, dtype=float).copy()
    bsubv_jxbout = np.asarray(bsubv, dtype=float).copy()
    if ns > 0:
        # VMEC jxbout only fills js=2..ns1; surface ns remains zero.
        bsubu_jxbout[-1] = 0.0
        bsubv_jxbout[-1] = 0.0

    np.savez(
        outdir / f"bsubs_ns{ns}_iter{int(iter_idx)}.npz",
        bsubs_half=np.asarray(bsubs_half, dtype=float),
        bsubs_full=np.asarray(bsubs_full, dtype=float),
        bsupu=np.asarray(bsupu, dtype=float),
        bsupv=np.asarray(bsupv, dtype=float),
        bsupu1=np.asarray(bsupu1, dtype=float),
        bsupv1=np.asarray(bsupv1, dtype=float),
        bsubu=np.asarray(bsubu, dtype=float),
        bsubv=np.asarray(bsubv, dtype=float),
        bsubu_jxbout=bsubu_jxbout,
        bsubv_jxbout=bsubv_jxbout,
        bsubu_raw=np.asarray(bsubu_raw, dtype=float),
        bsubv_raw=np.asarray(bsubv_raw, dtype=float),
        sqrtg=np.asarray(sqrtg, dtype=float),
        s=np.asarray(s, dtype=float),
    )


def _parse_jacobian_terms_dump(path: Path) -> dict[str, np.ndarray]:
    ns = ntheta = nzeta = None
    lines = path.read_text().splitlines()
    ru12 = zs = zu12 = rs = r12 = tau = rv12 = zv12 = None
    pru_e = pru_o = pz1_e = pz1_o = pzu_e = pzu_o = pr1_e = pr1_o = None
    prv_e = prv_o = pzv_e = pzv_o = None
    pshalf = None
    for line in lines:
        if not line or line[0] == "#":
            continue
        if line.startswith("ns="):
            ns = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("ntheta3="):
            ntheta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith("nzeta="):
            nzeta = int(line.split("=", 1)[-1].strip())
            continue
        if line.startswith(("ns=", "ntheta", "nzeta", "columns")):
            continue
        parts = line.split()
        if len(parts) < 3 + 33:
            continue
        if ns is None or ntheta is None or nzeta is None:
            continue
        if ru12 is None:
            ru12 = np.zeros((ns, ntheta, nzeta))
            zs = np.zeros_like(ru12)
            zu12 = np.zeros_like(ru12)
            rs = np.zeros_like(ru12)
            r12 = np.zeros_like(ru12)
            tau = np.zeros_like(ru12)
            rv12 = np.zeros_like(ru12)
            zv12 = np.zeros_like(ru12)
            pru_e = np.zeros_like(ru12)
            pru_o = np.zeros_like(ru12)
            pz1_e = np.zeros_like(ru12)
            pz1_o = np.zeros_like(ru12)
            pzu_e = np.zeros_like(ru12)
            pzu_o = np.zeros_like(ru12)
            pr1_e = np.zeros_like(ru12)
            pr1_o = np.zeros_like(ru12)
            prv_e = np.zeros_like(ru12)
            prv_o = np.zeros_like(ru12)
            pzv_e = np.zeros_like(ru12)
            pzv_o = np.zeros_like(ru12)
            pshalf = np.zeros_like(ru12)
        js = int(parts[0]) - 1
        lt = int(parts[1]) - 1
        lz = int(parts[2]) - 1
        # Column order from jacobian.f dump:
        # js lt lz pshalf
        # pru_e pru_o pru_e_m1 pru_o_m1
        # pz1_e pz1_o pz1_e_m1 pz1_o_m1
        # pzu_e pzu_o pzu_e_m1 pzu_o_m1
        # pr1_e pr1_o pr1_e_m1 pr1_o_m1
        # prv_e prv_o prv_e_m1 prv_o_m1
        # pzv_e pzv_o pzv_e_m1 pzv_o_m1
        # ru12 pzs pzu12 prs pr12 ptau rv12 zv12
        base = 3
        pshalf[js, lt, lz] = float(parts[base + 0])
        pru_e[js, lt, lz] = float(parts[base + 1])
        pru_o[js, lt, lz] = float(parts[base + 2])
        pz1_e[js, lt, lz] = float(parts[base + 5])
        pz1_o[js, lt, lz] = float(parts[base + 6])
        pzu_e[js, lt, lz] = float(parts[base + 9])
        pzu_o[js, lt, lz] = float(parts[base + 10])
        pr1_e[js, lt, lz] = float(parts[base + 13])
        pr1_o[js, lt, lz] = float(parts[base + 14])
        prv_e[js, lt, lz] = float(parts[base + 17])
        prv_o[js, lt, lz] = float(parts[base + 18])
        pzv_e[js, lt, lz] = float(parts[base + 21])
        pzv_o[js, lt, lz] = float(parts[base + 22])
        base2 = base + 1 + 4 + 4 + 4 + 4 + 4 + 4
        ru12[js, lt, lz] = float(parts[base2 + 0])
        zs[js, lt, lz] = float(parts[base2 + 1])
        zu12[js, lt, lz] = float(parts[base2 + 2])
        rs[js, lt, lz] = float(parts[base2 + 3])
        r12[js, lt, lz] = float(parts[base2 + 4])
        tau[js, lt, lz] = float(parts[base2 + 5])
        rv12[js, lt, lz] = float(parts[base2 + 6])
        zv12[js, lt, lz] = float(parts[base2 + 7])
    if ns is None or ntheta is None or nzeta is None:
        raise ValueError(f"Missing header fields in {path}")
    if ru12 is None:
        ru12 = np.zeros((ns, ntheta, nzeta))
        zs = np.zeros_like(ru12)
        zu12 = np.zeros_like(ru12)
        rs = np.zeros_like(ru12)
        r12 = np.zeros_like(ru12)
        tau = np.zeros_like(ru12)
        rv12 = np.zeros_like(ru12)
        zv12 = np.zeros_like(ru12)
        pru_e = np.zeros_like(ru12)
        pru_o = np.zeros_like(ru12)
        pz1_e = np.zeros_like(ru12)
        pz1_o = np.zeros_like(ru12)
        pzu_e = np.zeros_like(ru12)
        pzu_o = np.zeros_like(ru12)
        pr1_e = np.zeros_like(ru12)
        pr1_o = np.zeros_like(ru12)
        prv_e = np.zeros_like(ru12)
        prv_o = np.zeros_like(ru12)
        pzv_e = np.zeros_like(ru12)
        pzv_o = np.zeros_like(ru12)
        pshalf = np.zeros_like(ru12)
    return {
        "pshalf": pshalf,
        "pru_e": pru_e,
        "pru_o": pru_o,
        "pz1_e": pz1_e,
        "pz1_o": pz1_o,
        "pzu_e": pzu_e,
        "pzu_o": pzu_o,
        "pr1_e": pr1_e,
        "pr1_o": pr1_o,
        "prv_e": prv_e,
        "prv_o": prv_o,
        "pzv_e": pzv_e,
        "pzv_o": pzv_o,
        "ru12": ru12,
        "zs": zs,
        "zu12": zu12,
        "rs": rs,
        "r12": r12,
        "tau": tau,
        "rv12": rv12,
        "zv12": zv12,
    }


def _collect_bsube_terms_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsube_terms*_iter*.dat")):
        m = _RE_BSUBE_TERMS.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        data, ns_header = _parse_bsube_terms_dump(p)
        if ns_val is None:
            ns_val = ns_header
        out[(ns_val, it)] = data
    return out


def _collect_jacobian_terms_dumps(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    for p in sorted(path.glob("jacobian_terms_iter*.dat")):
        m = _RE_JACOBIAN_TERMS.match(p.name)
        if m:
            it = int(m.group(1))
            out[it] = _parse_jacobian_terms_dump(p)
    return out


def _collect_lulv_vmec_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lulv*_iter*.dat")):
        m = _RE_LULV.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_lulv_vmec_dump(p)
    return out


def _collect_lulv_jax_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lulv_ns*_iter*.npz")):
        m = _RE_LULV.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = _parse_lulv_jax_dump(p)
    return out


def _collect_lamcal_vmec_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, dict[str, np.ndarray]]]:
    out: dict[tuple[int | None, int], dict[str, dict[str, np.ndarray]]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lamcal_pre*_iter*.dat")):
        m = _RE_LAMCAL_PRE.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        out[(ns_val, it)] = {"pre": _parse_lamcal_vmec_dump(p)}
    for p in sorted(path.glob("lamcal_post*_iter*.dat")):
        m = _RE_LAMCAL_POST.search(p.name)
        if not m:
            continue
        ns_val, it = _parse_ns_iter(m, ns_group=1, iter_group=2)
        entry = out.setdefault((ns_val, it), {})
        entry["post"] = _parse_lamcal_vmec_dump(p)
    return out


def _collect_lamcal_jax_dumps(path: Path) -> dict[tuple[int | None, int], dict[str, np.ndarray]]:
    out: dict[tuple[int | None, int], dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("lamcal_ns*_iter*.npz")):
        m = re.search(r"lamcal_ns(\d+)_iter(\d+)\.npz$", p.name)
        if not m:
            continue
        ns_val = int(m.group(1))
        it = int(m.group(2))
        out[(ns_val, it)] = _parse_lamcal_jax_dump(p)
    return out


T = TypeVar("T")


def _lookup_by_ns(dumps: dict[tuple[int | None, int], T], *, ns: int, it: int) -> T | None:
    if (ns, it) in dumps:
        return dumps[(ns, it)]
    if (None, it) in dumps:
        return dumps[(None, it)]
    return None


def _resolve_other(
    dumps: dict[tuple[int | None, int], T],
    *,
    ns: int | None,
    it: int,
) -> tuple[int | None, T | None]:
    if ns is not None and (ns, it) in dumps:
        return ns, dumps[(ns, it)]
    if (None, it) in dumps:
        return None, dumps[(None, it)]
    for (ns_k, it_k), val in dumps.items():
        if it_k == it:
            return ns_k, val
    return None, None


def _compare_vectors(
    *,
    label: str,
    vmec_vec: np.ndarray,
    jax_vec: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[bool, str, int]:
    if vmec_vec.shape != jax_vec.shape:
        return False, f"{label}: shape mismatch vmec={vmec_vec.shape} jax={jax_vec.shape}", -1
    diff = np.abs(vmec_vec - jax_vec)
    if diff.size == 0:
        return True, f"{label}: empty", -1
    i = int(np.argmax(diff))
    max_abs = float(diff[i])
    denom = max(float(atol), float(abs(vmec_vec[i])))
    max_rel = max_abs / denom if denom != 0.0 else float("inf")
    ok = max_abs <= max(float(atol), float(rtol) * abs(vmec_vec[i]))
    msg = f"{label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={i}"
    return ok, msg, i


def _format_kernel_index(idx: int, *, shape: tuple[int, int, int, int]) -> str:
    try:
        js, lt, lz, mpar = np.unravel_index(idx, shape)
    except Exception:
        return f"idx={idx}"
    return f"js={js+1} lt={lt+1} lz={lz+1} mpar={mpar}"


def _radial_keep_mask_flat(*, size: int, ns: int, radial_skip: int, radial_drop_edge: bool) -> np.ndarray:
    """Flat-vector mask that keeps only selected radial indices (js-fast layout)."""
    size = int(size)
    ns = int(ns)
    if size <= 0:
        return np.zeros((0,), dtype=bool)
    if ns <= 0:
        return np.ones((size,), dtype=bool)
    js = np.arange(size, dtype=np.int64) % ns
    keep = js >= max(int(radial_skip), 0)
    if bool(radial_drop_edge):
        keep = keep & (js < (ns - 1))
    return keep


def _decode_xc_index(idx: int, *, ns: int, mpol: int, ntor: int, lthreed: bool, lasym: bool = False) -> str:
    """Decode xc/xcdot flat index into (component, m, n, js) info."""
    ns = int(ns)
    mpol = int(mpol)
    ntor = int(ntor)
    lthreed = bool(lthreed)
    lasym = bool(lasym)
    ntmax = (2 if lthreed else 1) * (2 if lasym else 1)
    nrange = ntor + 1
    mnsize = mpol * nrange
    mns = ns * mnsize
    if mns <= 0:
        return "idx decode unavailable (mns=0)"
    ntype = idx // mns
    if ntype >= 3 * ntmax:
        return f"idx={idx} out of range for mns={mns} ntmax={ntmax}"
    offset = idx - ntype * mns
    mn = offset // ns
    js = offset - mn * ns
    m = mn // nrange
    n = mn - m * nrange
    if (not lthreed) and (not lasym):
        comps = ("rcc", "zsc", "lsc")
    elif (not lthreed) and lasym:
        # VMEC serial packed layout (LTHREED=F, LASYM=T):
        # [rcc, rsc, zsc, zcc, lsc, lcc]
        comps = ("rcc", "rsc", "zsc", "zcc", "lsc", "lcc")
    elif lthreed and (not lasym):
        # VMEC serial packed layout (LTHREED=T, LASYM=F):
        # [rcc, rss, zsc, zcs, lsc, lcs]
        comps = ("rcc", "rss", "zsc", "zcs", "lsc", "lcs")
    else:
        # VMEC serial packed layout (LTHREED=T, LASYM=T):
        # [rcc, rss, rsc, rcs, zsc, zcs, zcc, zss, lsc, lcs, lcc, lss]
        comps = ("rcc", "rss", "rsc", "rcs", "zsc", "zcs", "zcc", "zss", "lsc", "lcs", "lcc", "lss")
    comp = comps[ntype] if 0 <= ntype < len(comps) else f"ntype={ntype}"
    return f"{comp}: m={m} n={n} js={js + 1} (ns={ns})"


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


def _focus_rel_stats(
    ref_vals: np.ndarray,
    test_vals: np.ndarray,
    *,
    focus_abs: float,
    eps: float = 1e-30,
) -> tuple[float, float, float, int, int] | None:
    ref = np.asarray(ref_vals, dtype=float)
    test = np.asarray(test_vals, dtype=float)
    mask = np.isfinite(ref) & np.isfinite(test) & (np.abs(ref) > float(focus_abs))
    total = int(ref.size)
    n = int(np.count_nonzero(mask))
    if n <= 0:
        return None
    diff = np.abs(test[mask] - ref[mask])
    ref_abs = np.maximum(np.abs(ref[mask]), float(eps))
    rel = diff / ref_abs
    max_abs = float(np.max(diff))
    max_rel = float(np.max(rel))
    rms_rel = float(np.sqrt(np.mean(rel * rel)))
    return max_abs, max_rel, rms_rel, n, total


def _decode_tomnsps_index(idx: int, shape: tuple[int, int, int]) -> str:
    ns, mpol1p1, ntorp1 = shape
    if idx < 0 or ns <= 0 or mpol1p1 <= 0 or ntorp1 <= 0:
        return "idx decode unavailable"
    js = idx // (mpol1p1 * ntorp1)
    rem = idx % (mpol1p1 * ntorp1)
    m = rem // ntorp1
    n = rem % ntorp1
    return f"js={js+1} m={m} n={n}"


def _decode_gc_index(idx: int, shape: tuple[int, int, int, int]) -> str:
    ns, ntorp1, mpol1p1, ntmax = shape
    if idx < 0 or ns <= 0 or ntorp1 <= 0 or mpol1p1 <= 0 or ntmax <= 0:
        return "idx decode unavailable"
    js = idx // (ntorp1 * mpol1p1 * ntmax)
    rem = idx % (ntorp1 * mpol1p1 * ntmax)
    n = rem // (mpol1p1 * ntmax)
    rem2 = rem % (mpol1p1 * ntmax)
    m = rem2 // ntmax
    t = rem2 % ntmax
    return f"js={js+1} n={n} m={m} t={t+1}"


def _rms(x: np.ndarray, *, eps: float = 1e-30) -> float:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    val = float(np.sqrt(np.mean(x * x)))
    if not np.isfinite(val):
        return 0.0
    return max(val, eps)


def _fsql1_from_gcl(gcl: np.ndarray, delta_s: float) -> float:
    if gcl is None:
        return 0.0
    gcl = np.asarray(gcl, dtype=float)
    if gcl.size == 0 or gcl.ndim == 0:
        return 0.0
    gcl_use = gcl[1:] if gcl.shape[0] > 1 else gcl
    return float(delta_s) * float(np.sum(gcl_use * gcl_use))


def _align_lam_arrays(vm: np.ndarray, jx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vm = np.asarray(vm, dtype=float)
    jx = np.asarray(jx, dtype=float)
    if jx.ndim == 3:
        # (ns, m, n) -> (ns, n, m, 1)
        jx = np.transpose(jx, (0, 2, 1))[:, :, :, None]
    if jx.ndim == 4 and vm.ndim == 4:
        if jx.shape[3] == 1 and vm.shape[3] > 1:
            jx = np.repeat(jx, vm.shape[3], axis=3)
    if vm.ndim != 4 or jx.ndim != 4:
        return vm, jx
    ns_min = min(vm.shape[0], jx.shape[0])
    n_min = min(vm.shape[1], jx.shape[1])
    m_min = min(vm.shape[2], jx.shape[2])
    t_min = min(vm.shape[3], jx.shape[3])
    vm = vm[:ns_min, :n_min, :m_min, :t_min]
    jx = jx[:ns_min, :n_min, :m_min, :t_min]
    return vm, jx


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

        for k_up, pat in key_re.items():
            if pat.match(line):
                indent = pat.match(line).group(1)
                lines[i] = f"{indent}{k_up} = {updates[k_up]}"
                found[k_up] = True
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

    if end_idx is None:
        if not in_block:
            return text
        end_idx = len(lines)

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
        "--split-iter",
        type=int,
        default=0,
        help="If >0, run vmec_jax in two phases (split_iter + remaining) with warm start.",
    )
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
        "--ns-array",
        type=str,
        default=None,
        help="Override NS_ARRAY staging (space/comma separated list).",
    )
    p.add_argument(
        "--niter-array",
        type=str,
        default=None,
        help="Override NITER_ARRAY staging (space/comma separated list).",
    )
    p.add_argument(
        "--ftol-array",
        type=str,
        default=None,
        help="Override FTOL_ARRAY staging (space/comma separated list).",
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
        "--fsq-relax-threshold",
        type=float,
        default=1.0e-30,
        help=(
            "Relax fsq* fail-fast tolerance when |VMEC fsq*| exceeds this value. "
            "Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--fsq-relax-rtol",
        type=float,
        default=2.0e-2,
        help=(
            "Relaxed rtol for fsq/fsq1 fields above --fsq-relax-threshold."
        ),
    )
    p.add_argument(
        "--fsq1-relax-threshold",
        type=float,
        default=1.0e-30,
        help=(
            "Relax fsq1 (preconditioned) fail-fast tolerance when |VMEC fsq1| "
            "exceeds this value. Set <=0 to disable."
        ),
    )
    p.add_argument(
        "--radial-skip",
        type=int,
        default=6,
        help="Number of leading radial points to skip in end-state wout radial parity checks.",
    )
    p.add_argument(
        "--radial-drop-edge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also drop the last radial point in end-state wout radial parity checks.",
    )
    p.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero at the first mismatch beyond tolerances (default: True).",
    )
    p.add_argument(
        "--dump-level",
        choices=("full", "lite", "none"),
        default="full",
        help=(
            "Control VMEC2000 dump verbosity. "
            "'full' enables all dumps; 'lite' keeps scalar/trace dumps only; "
            "'none' disables VMEC/JAX dumps."
        ),
    )
    p.add_argument(
        "--dump-iter",
        type=str,
        default=None,
        help=(
            "Optional comma-separated iteration list/ranges to limit dumps "
            "(passed through to VMEC_DUMP_ITER / VMEC_JAX_DUMP_ITER, e.g. '1,3-5')."
        ),
    )
    p.add_argument(
        "--fsq-from-dumps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use VMEC2000 scalar dumps to override threed1 fsq* columns.",
    )
    p.add_argument(
        "--vmec-timeout",
        type=float,
        default=None,
        help="Timeout (seconds) for the VMEC2000 run. Default: no timeout.",
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Optional work directory for VMEC2000/vmec_jax runs (preserves dumps).",
    )
    args = p.parse_args()
    vmec_fsq_dump: dict[tuple[int | None, int], dict[str, float]] = {}
    jax_fsq_dump: dict[tuple[int | None, int], dict[str, float]] = {}
    vmec_fsq1: dict[tuple[int | None, int], dict[str, float]] = {}
    vmec_time_control: dict[int, float] = {}

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
    ns_override = _parse_int_list_arg(args.ns_array)
    niter_override = _parse_int_list_arg(args.niter_array)
    ftol_override = _parse_float_list_arg(args.ftol_array)
    use_input_niter_flag = bool(args.use_input_niter) or bool(ns_override or niter_override or ftol_override)
    if args.single_ns is None:
        ns_stages_eff, niter_stages_eff, ftol_stages_eff = _resolve_stage_controls(
            cfg=_cfg_in,
            indata=_indata_in,
            max_iter=int(args.max_iter),
            use_input_niter=use_input_niter_flag,
        )
        if ns_override is not None:
            ns_stages_eff = ns_override
        if niter_override is not None:
            niter_stages_eff = niter_override
        if ftol_override is not None:
            ftol_stages_eff = ftol_override
        if ns_stages_eff is not None:
            stage_len = len(ns_stages_eff)
            if niter_stages_eff is None:
                niter_stages_eff = [int(args.max_iter)] * stage_len
            else:
                niter_stages_eff = _extend_list([int(v) for v in niter_stages_eff], stage_len)
            if ftol_stages_eff is None:
                ftol_stages_eff = [ftol_default] * stage_len
            else:
                ftol_stages_eff = _extend_list([float(v) for v in ftol_stages_eff], stage_len)
    elif ns_override is not None or niter_override is not None or ftol_override is not None:
        raise SystemExit("--single-ns cannot be combined with --ns-array/--niter-array/--ftol-array overrides.")

    # --- Run VMEC2000 executable in an isolated workdir ---
    threed1_stages: list[Vmec2000Threed1Stage] | None = None
    workdir_arg = Path(args.workdir).expanduser().resolve() if args.workdir is not None else None
    with _workdir_context(workdir_arg) as workdir:
        input_local = _copy_input_and_mgrid(input_path, workdir)
        vmec_dump_dir = workdir / "vmec_dumps"
        jax_dump_dir = workdir / "jax_dumps"
        vmec_dump_dir.mkdir(parents=True, exist_ok=True)
        jax_dump_dir.mkdir(parents=True, exist_ok=True)

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
        jax_input_path = input_local
        cmd = [str(vmec2000_exe), input_local.name]
        vmec_env = os.environ.copy()
        if args.dump_level != "none":
            vmec_env["VMEC_DUMP_DIR"] = str(vmec_dump_dir)
            vmec_env["VMEC_DUMP_SCALARS"] = "1"
            vmec_env["VMEC_DUMP_GCX2"] = "1"
            vmec_env["VMEC_DUMP_FSQ1"] = "1"
            vmec_env["VMEC_DUMP_TIMECONTROL"] = "1"
            if args.dump_level == "full":
                vmec_env["VMEC_DUMP_XC"] = "1"
                vmec_env["VMEC_DUMP_BSUBE"] = "1"
                vmec_env["VMEC_DUMP_BSUBE_TERMS"] = "1"
                vmec_env["VMEC_DUMP_BSUP"] = "1"
                vmec_env["VMEC_DUMP_BSUBH"] = "1"
                vmec_env["VMEC_DUMP_BSUB_PRE_SYM"] = "1"
                vmec_env["VMEC_DUMP_WROUT_MODES"] = "1"
                vmec_env["VMEC_DUMP_BSUBS"] = "1"
                vmec_env["VMEC_DUMP_BSS_INPUTS"] = "1"
                vmec_env["VMEC_DUMP_BSS_TERMS"] = "1"
                vmec_env["VMEC_DUMP_JXB_CHANNELS"] = "1"
                vmec_env["VMEC_DUMP_JACOBIAN_TERMS"] = "1"
                vmec_env["VMEC_DUMP_LULV"] = "1"
                vmec_env["VMEC_DUMP_LAMCAL"] = "1"
                vmec_env["VMEC_DUMP_TOMNSPS"] = "1"
                vmec_env["VMEC_DUMP_TOMNSPS_KERNELS"] = "1"
                vmec_env["VMEC_DUMP_GC"] = "1"
                vmec_env["VMEC_DUMP_GC_STAGE"] = "both"
                vmec_env["VMEC_DUMP_GC_DIR"] = str(vmec_dump_dir)
                vmec_env["VMEC_DUMP_LAM"] = "1"
                vmec_env["VMEC_DUMP_PRECOND"] = "1"
                vmec_env["VMEC_DUMP_BCOVAR"] = "1"
                vmec_env["VMEC_DUMP_CONSTRAINTS"] = "1"
            if args.dump_iter:
                vmec_env["VMEC_DUMP_ITER"] = str(args.dump_iter)
                vmec_env["VMEC_DUMP_XC_ITER"] = str(args.dump_iter)
            else:
                vmec_env.pop("VMEC_DUMP_ITER", None)
                vmec_env.pop("VMEC_DUMP_XC_ITER", None)
        else:
            for key in list(vmec_env.keys()):
                if key.startswith("VMEC_DUMP_"):
                    vmec_env.pop(key, None)
        t0_vmec = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
                env=vmec_env,
                timeout=args.vmec_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"VMEC2000 run timed out after {args.vmec_timeout}s") from exc
        vmec_time_s = time.perf_counter() - t0_vmec
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
        wout = None
        if wout_path.exists():
            try:
                wout = read_wout(wout_path)
            except Exception as exc:
                print(f"[warn] failed to read VMEC2000 wout {wout_path}: {exc}")
        jxbout_path = workdir / f"jxbout_{suffix}.nc"
        vmec_jxbout = _read_jxbout(jxbout_path)

        # --- Run vmec_jax with VMEC-style multigrid staging ---
        # Some VMEC2000 builds remove the input file from the workdir.
        # Re-copy it to ensure vmec_jax can read the same input.
        if not input_local.exists():
            shutil.copy2(input_path, input_local)
        jax_input_path = input_local
        def _run_vmec_jax(*, dump_dir: Path, max_iter: int, restart_state=None, restart_solver_state=None, multigrid: bool | None = None):
            jax_env_backup = os.environ.copy()
            if args.dump_level != "none":
                os.environ["VMEC_JAX_DUMP_DIR"] = str(dump_dir)
                os.environ["VMEC_JAX_DUMP_SCALARS"] = "1"
                os.environ["VMEC_JAX_DUMP_GCX2"] = "1"
                os.environ["VMEC_JAX_DUMP_TIMECONTROL"] = "1"
                if args.dump_level == "full":
                    os.environ["VMEC_JAX_DUMP_XC"] = "1"
                    os.environ["VMEC_JAX_DUMP_XC_INIT"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSUBE"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSUBE_TERMS"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSUBH"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSUB_PRE_SYM"] = "1"
                    os.environ["VMEC_JAX_DUMP_WROUT_MODES"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSUBS"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSS_INPUTS"] = "1"
                    os.environ["VMEC_JAX_DUMP_BSS_TERMS"] = "1"
                    os.environ["VMEC_JAX_DUMP_JXB_CHANNELS"] = "1"
                    os.environ["VMEC_JAX_DUMP_LULV"] = "1"
                    os.environ["VMEC_JAX_DUMP_LAMCAL"] = "1"
                    os.environ["VMEC_JAX_DUMP_TOMNSPS"] = "1"
                    os.environ["VMEC_JAX_DUMP_FORCE_KERNELS"] = "1"
                    os.environ["VMEC_JAX_DUMP_GC"] = "1"
                    os.environ["VMEC_JAX_DUMP_GC_STAGE"] = "both"
                    os.environ["VMEC_JAX_DUMP_GC_DIR"] = str(dump_dir)
                    os.environ["VMEC_JAX_DUMP_LAM"] = "1"
                    os.environ["VMEC_JAX_DUMP_JACOBIAN_TERMS"] = "1"
                    os.environ["VMEC_JAX_DUMP_BCOVAR"] = "1"
                    os.environ["VMEC_JAX_DUMP_CONSTRAINTS"] = "1"
                if args.dump_iter:
                    os.environ["VMEC_JAX_DUMP_ITER"] = str(args.dump_iter)
                else:
                    os.environ.pop("VMEC_JAX_DUMP_ITER", None)
            else:
                for key in list(os.environ.keys()):
                    if key.startswith("VMEC_JAX_DUMP_"):
                        os.environ.pop(key, None)
            if args.vmec_nstep is not None:
                os.environ["VMEC_JAX_NSTEP_OVERRIDE"] = str(int(args.vmec_nstep))
            try:
                # Parity runs should use the non-scan path to match VMEC2000.
                # The scan path is performance-focused and can diverge for large ns.
                use_scan = False
                performance_mode = False
                return vj.run_fixed_boundary(
                    jax_input_path,
                    solver="vmec2000_iter",
                    max_iter=int(max_iter),
                    multigrid_use_input_niter=use_input_niter_flag,
                    multigrid=multigrid,
                    verbose=False,
                    use_scan=use_scan,
                    performance_mode=performance_mode,
                    ns_override=int(args.single_ns) if args.single_ns is not None else None,
                    restart_state=restart_state,
                    restart_solver_state=restart_solver_state,
                )
            finally:
                os.environ.clear()
                os.environ.update(jax_env_backup)

        split_iter = int(args.split_iter)
        resume_state = None
        use_split = split_iter > 0 and split_iter < int(args.max_iter)
        run2 = None
        jax_time_s = 0.0
        if use_split:
            jax_dump_dir1 = jax_dump_dir / "phase1"
            jax_dump_dir2 = jax_dump_dir / "phase2"
            jax_dump_dir1.mkdir(parents=True, exist_ok=True)
            jax_dump_dir2.mkdir(parents=True, exist_ok=True)
            t0_jax = time.perf_counter()
            run1 = _run_vmec_jax(dump_dir=jax_dump_dir1, max_iter=int(split_iter), restart_state=None, multigrid=None)
            jax_time_s += time.perf_counter() - t0_jax
            if run1.result is not None:
                resume_state = run1.result.diagnostics.get("resume_state")
            offset_iter = int(np.asarray(run1.result.w_history).size) if run1.result is not None else int(split_iter)
            remaining = int(args.max_iter) - int(offset_iter)
            if remaining > 0:
                t0_jax = time.perf_counter()
                run2 = _run_vmec_jax(
                    dump_dir=jax_dump_dir2,
                    max_iter=int(remaining),
                    restart_state=run1.state,
                    restart_solver_state=resume_state,
                    multigrid=False,
                )
                jax_time_s += time.perf_counter() - t0_jax
            run = run1 if run2 is None else vj.FixedBoundaryRun(
                cfg=run2.cfg,
                indata=run2.indata,
                static=run2.static,
                state=run2.state,
                result=_merge_vmec_results(run1.result, run2.result),
                flux=run2.flux,
                profiles=run2.profiles,
                signgs=run2.signgs,
            )
        else:
            t0_jax = time.perf_counter()
            run = _run_vmec_jax(dump_dir=jax_dump_dir, max_iter=int(args.max_iter))
            jax_time_s += time.perf_counter() - t0_jax

        print(f"Runtime (wall): vmec2000={vmec_time_s:.3f}s  vmec_jax={jax_time_s:.3f}s")

        # Ensure we have a final-state bsubs dump for jxbout parity, even if
        # per-iteration dumps were filtered out.
        try:
            if use_split:
                final_dump_dir = jax_dump_dir2 if run2 is not None else jax_dump_dir1
            else:
                final_dump_dir = jax_dump_dir
            _dump_bsubs_final(run, final_dump_dir)
        except Exception:
            pass

        vmec_xc = _collect_vmec_xc_dumps(vmec_dump_dir)
        dump_offset = int(np.asarray(run1.result.w_history).size) if (use_split and run1 is not None and run1.result is not None) else int(split_iter)
        dump_overwrite = True
        if use_split and run2 is not None and resume_state is not None:
            resume_offset = int(resume_state.get("iter_offset", 0))
            if resume_offset > 0:
                dump_offset = 0
                dump_overwrite = False
        if use_split:
            jax_dump_dir_active = jax_dump_dir1
        else:
            jax_dump_dir_active = jax_dump_dir
        if use_split and run2 is not None:
            jax_xc = _merge_dump_dicts(
                _collect_jax_xc_dumps(jax_dump_dir1),
                _collect_jax_xc_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_xc = _collect_jax_xc_dumps(jax_dump_dir_active)
        if use_split and run2 is not None:
            jax_xc_init = {**_collect_jax_xc_init_dumps(jax_dump_dir1)}
            for k, v in _collect_jax_xc_init_dumps(jax_dump_dir2).items():
                jax_xc_init.setdefault(k, v)
        else:
            jax_xc_init = _collect_jax_xc_init_dumps(jax_dump_dir_active)
        vmec_bsube = _collect_bsube_dumps(vmec_dump_dir)
        vmec_bsube_terms = _collect_bsube_terms_dumps(vmec_dump_dir)
        vmec_bsubs = _collect_vmec_bsubs_dumps(vmec_dump_dir)
        vmec_jacobian_terms = _collect_jacobian_terms_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_bsube = _merge_dump_dicts(
                _collect_bsube_dumps(jax_dump_dir1),
                _collect_bsube_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_bsube_terms = _merge_dump_dicts(
                _collect_bsube_terms_dumps(jax_dump_dir1),
                _collect_bsube_terms_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_bsubs = _merge_dump_dicts(
                _collect_bsubs_dumps(jax_dump_dir1),
                _collect_bsubs_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_bsubh = _merge_dump_dicts(
                _collect_bsubh_dumps(jax_dump_dir1),
                _collect_bsubh_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_bsub_pre = _merge_dump_dicts(
                _collect_bsub_pre_sym_dumps(jax_dump_dir1),
                _collect_bsub_pre_sym_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_bsube = _collect_bsube_dumps(jax_dump_dir_active)
            jax_bsube_terms = _collect_bsube_terms_dumps(jax_dump_dir_active)
            jax_bsubs = _collect_bsubs_dumps(jax_dump_dir_active)
            jax_bsubh = _collect_bsubh_dumps(jax_dump_dir_active)
            jax_bsub_pre = _collect_bsub_pre_sym_dumps(jax_dump_dir_active)
        vmec_bsubh = _collect_vmec_bsubh_dumps(vmec_dump_dir)
        vmec_bsub_pre = _collect_vmec_bsub_pre_sym_dumps(vmec_dump_dir)
        vmec_bss_inputs = _collect_vmec_bss_inputs(vmec_dump_dir)
        vmec_bss_terms = _collect_vmec_bss_terms(vmec_dump_dir)
        jax_bss_dir = jax_dump_dir_active
        if use_split and run2 is not None:
            jax_bss_dir = jax_dump_dir2
        jax_bss_inputs = _collect_jax_bss_inputs(jax_bss_dir)
        jax_bss_terms = _collect_jax_bss_terms(jax_bss_dir)
        vmec_lulv = _collect_lulv_vmec_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_lulv = _merge_dump_dicts(
                _collect_lulv_jax_dumps(jax_dump_dir1),
                _collect_lulv_jax_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_lulv = _collect_lulv_jax_dumps(jax_dump_dir_active)
        vmec_gc = _collect_vmec_gc_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_gc = _merge_dump_dicts(
                _collect_jax_gc_dumps(jax_dump_dir1),
                _collect_jax_gc_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_gc = _collect_jax_gc_dumps(jax_dump_dir_active)
        vmec_tomnsps = _collect_vmec_tomnsps_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_tomnsps = _merge_dump_dicts(
                _collect_jax_tomnsps_dumps(jax_dump_dir1),
                _collect_jax_tomnsps_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_tomnsps = _collect_jax_tomnsps_dumps(jax_dump_dir_active)
        vmec_kernels = _collect_vmec_tomnsps_kernels_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_kernels = _merge_dump_dicts(
                _collect_jax_force_kernels(jax_dump_dir1),
                _collect_jax_force_kernels(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_kernels = _collect_jax_force_kernels(jax_dump_dir_active)
        vmec_lam = _collect_vmec_lam_dumps(vmec_dump_dir)
        vmec_lam_fsql1 = _collect_lam_fsql1_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_lam = _merge_dump_dicts(
                _collect_jax_lam_dumps(jax_dump_dir1),
                _collect_jax_lam_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_lam_fsql1 = _merge_dump_dicts(
                _collect_lam_fsql1_dumps(jax_dump_dir1),
                _collect_lam_fsql1_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
            jax_lam_gcl = _merge_dump_dicts(
                _collect_lam_gcl_dumps(jax_dump_dir1),
                _collect_lam_gcl_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_lam = _collect_jax_lam_dumps(jax_dump_dir_active)
            jax_lam_fsql1 = _collect_lam_fsql1_dumps(jax_dump_dir_active)
            jax_lam_gcl = _collect_lam_gcl_dumps(jax_dump_dir_active)
        vmec_lamcal = _collect_lamcal_vmec_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_lamcal = _merge_dump_dicts(
                _collect_lamcal_jax_dumps(jax_dump_dir1),
                _collect_lamcal_jax_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_lamcal = _collect_lamcal_jax_dumps(jax_dump_dir_active)
        vmec_scalars = _collect_scalars_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_scalars = _merge_dump_dicts(
                _collect_scalars_dumps(jax_dump_dir1),
                _collect_scalars_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_scalars = _collect_scalars_dumps(jax_dump_dir_active)
        vmec_gcx2 = _collect_gcx2_dumps(vmec_dump_dir)
        if use_split and run2 is not None:
            jax_gcx2 = _merge_dump_dicts(
                _collect_gcx2_dumps(jax_dump_dir1),
                _collect_gcx2_dumps(jax_dump_dir2),
                offset=dump_offset,
                overwrite=dump_overwrite,
            )
        else:
            jax_gcx2 = _collect_gcx2_dumps(jax_dump_dir_active)
        vmec_fsq1 = _collect_fsq1_dumps(vmec_dump_dir)
        vmec_time_control = _parse_time_control_trace(vmec_dump_dir / "time_control_trace.log")
        if getattr(run, "static", None) is not None and getattr(run.static, "trig", None) is not None:
            r0scale = float(run.static.trig.r0scale)
        else:
            r0scale = 1.0
        r1 = 1.0 / (2.0 * r0scale) ** 2
        vmec_fsq_dump = _compute_fsq_from_dumps(scalars=vmec_scalars, gcx2=vmec_gcx2, r1=r1)
        jax_fsq_dump = _compute_fsq_from_dumps(scalars=jax_scalars, gcx2=jax_gcx2, r1=r1)

    # --- Report ---
    use_threed1 = bool(threed1_stages)
    vmec_stages = threed1_stages if use_threed1 else stages
    vmec_ns = np.asarray([int(st.ns) for st in vmec_stages], dtype=int)
    vmec_niter = np.asarray(
        [int(len(st.rows)) if st.rows else int(st.niter) for st in vmec_stages],
        dtype=int,
    )
    vmec_offsets = np.concatenate([[0], np.cumsum(vmec_niter[:-1])]).astype(int) if vmec_niter.size else np.zeros((0,), dtype=int)

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
    include_edge_hist = np.asarray(diag.get("include_edge_history", np.zeros((0,), dtype=int)), dtype=int)
    zero_m1_hist = np.asarray(diag.get("zero_m1_history", np.zeros((0,), dtype=int)), dtype=int)

    if isinstance(diag, dict) and diag:
        bcovar_hist = np.asarray(diag.get("bcovar_update_history", np.zeros((0,), dtype=int)), dtype=int)
        restart_hist = np.asarray(diag.get("restart_path_history", np.zeros((0,), dtype=object)), dtype=object)
        time_hist = np.asarray(diag.get("time_step_history", np.zeros((0,), dtype=float)), dtype=float)
        if bcovar_hist.size or restart_hist.size:
            nshow = int(min(10, max(bcovar_hist.size, restart_hist.size, time_hist.size)))
            def _fmt(arr):
                return ", ".join(str(v) for v in arr[:nshow])
            print()
            print("vmec_jax cadence (first 10 iters):")
            if time_hist.size:
                print(f"  time_step_history: [{_fmt(time_hist)}]")
            if bcovar_hist.size:
                print(f"  bcovar_update_history: [{_fmt(bcovar_hist)}]")
            if restart_hist.size:
                print(f"  restart_path_history: [{_fmt(restart_hist)}]")

    print()
    if use_threed1:
        if vmec_fsq_dump:
            print("Stage/iter comparison (VMEC2000 fsq dumps + threed1 scalars vs vmec_jax histories):")
        else:
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
    first_mismatch: dict[str, object] | None = None
    stop_after_mismatch = False
    if use_threed1:
        for name in ("fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1", "delt0r", "r00", "w"):
            diff_cols_vmec[name] = []
            diff_cols_jax[name] = []

    def _matches(
        vmec_val: float,
        jax_val: float,
        *,
        field: str | None = None,
        round_step: float | None = None,
    ) -> bool:
        if not (np.isfinite(vmec_val) and np.isfinite(jax_val)):
            return False
        rtol_eff = float(args.rtol)
        if field in {"fsqr", "fsqz", "fsql", "fsqr1", "fsqz1", "fsql1"}:
            if field in {"fsqr1", "fsqz1", "fsql1"}:
                thr = float(args.fsq1_relax_threshold)
            else:
                thr = float(args.fsq_relax_threshold)
            if (thr > 0.0) and (abs(float(vmec_val)) >= thr):
                rtol_eff = max(rtol_eff, float(args.fsq_relax_rtol))
        tol = max(float(args.atol), rtol_eff * abs(vmec_val))
        if round_step is not None and np.isfinite(round_step):
            tol = max(tol, float(round_step))
        return abs(vmec_val - jax_val) <= tol

    def _round_step(val: float, decimals: int) -> float:
        if not np.isfinite(val) or val == 0.0:
            return 0.0
        exp = math.floor(math.log10(abs(val)))
        return (10.0 ** (exp - int(decimals))) * 1.01

    def _vmec_print_round(val: float, decimals: int) -> float:
        if not np.isfinite(val):
            return val
        try:
            return float(f"{float(val):.{int(decimals)}E}")
        except Exception:
            return float(val)

    # Stage transition parity (ns + offsets).
    if offsets.size and ns_stages.size:
        ns_ok = bool(vmec_ns.size == ns_stages.size) and bool(np.all(vmec_ns == ns_stages[: vmec_ns.size]))
        off_ok = bool(vmec_offsets.size == offsets.size) and bool(np.all(vmec_offsets == offsets[: vmec_offsets.size]))
        if not ns_ok or not off_ok:
            print()
            print("Stage transition mismatch:")
            if not ns_ok:
                print(f"  vmec ns_stages={vmec_ns.tolist()}  jax ns_stages={ns_stages.tolist()}")
            if not off_ok:
                print(f"  vmec offsets={vmec_offsets.tolist()}  jax offsets={offsets.tolist()}")
            if bool(args.fail_fast) and first_mismatch is None:
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
                vmec_fsqr = float(row.fsqr)
                vmec_fsqz = float(row.fsqz)
                vmec_fsql = float(row.fsql)
                vmec_fsqr1 = float(row.fsqr1)
                vmec_fsqz1 = float(row.fsqz1)
                vmec_fsql1 = float(row.fsql1)
                vmec_delt0r = float(row.delt0r if row.delt0r is not None else float("nan"))
                key_ns = int(st.ns)
                key_it = int(row.it)
                if args.fsq_from_dumps and vmec_fsq_dump:
                    dump_vals = _lookup_by_ns(vmec_fsq_dump, ns=key_ns, it=key_it)
                    if dump_vals is not None:
                        vmec_fsqr = float(dump_vals.get("fsqr", vmec_fsqr))
                        vmec_fsqz = float(dump_vals.get("fsqz", vmec_fsqz))
                        vmec_fsql = float(dump_vals.get("fsql", vmec_fsql))
                if args.fsq_from_dumps and vmec_fsq1:
                    dump_vals = _lookup_by_ns(vmec_fsq1, ns=key_ns, it=key_it)
                    if dump_vals is not None:
                        vmec_fsqr1 = float(dump_vals.get("fsqr1", vmec_fsqr1))
                        vmec_fsqz1 = float(dump_vals.get("fsqz1", vmec_fsqz1))
                        vmec_fsql1 = float(dump_vals.get("fsql1", vmec_fsql1))
                if vmec_time_control and not math.isfinite(vmec_delt0r):
                    iter2 = off + int(row.it)
                    if iter2 in vmec_time_control:
                        vmec_delt0r = float(vmec_time_control[iter2])
                    else:
                        max_key = max(vmec_time_control.keys(), default=None)
                        if max_key is not None and iter2 > max_key:
                            vmec_delt0r = float(vmec_time_control[max_key])
                jx_fsqr = float(fsqr[j] if j < fsqr.size else float("nan"))
                jx_fsqz = float(fsqz[j] if j < fsqz.size else float("nan"))
                jx_fsql = float(fsql[j] if j < fsql.size else float("nan"))
                jx_fsqr1 = float(fsqr1[j] if j < fsqr1.size else float("nan"))
                jx_fsqz1 = float(fsqz1[j] if j < fsqz1.size else float("nan"))
                jx_fsql1 = float(fsql1[j] if j < fsql1.size else float("nan"))
                jx_delt0r = float(delt[j] if j < delt.size else float("nan"))
                jx_r00 = float(r00[j] if j < r00.size else float("nan"))
                jx_w = float(w[j] if j < w.size else float("nan"))
                # When comparing to threed1 (no dumps), round to VMEC print precision
                # so the comparator reflects what is actually printed.
                use_dump_vals = bool(args.fsq_from_dumps) and (bool(vmec_fsq_dump) or bool(vmec_fsq1))
                if not use_dump_vals:
                    vmec_fsqr = _vmec_print_round(vmec_fsqr, 2)
                    vmec_fsqz = _vmec_print_round(vmec_fsqz, 2)
                    vmec_fsql = _vmec_print_round(vmec_fsql, 2)
                    vmec_fsqr1 = _vmec_print_round(vmec_fsqr1, 2)
                    vmec_fsqz1 = _vmec_print_round(vmec_fsqz1, 2)
                    vmec_fsql1 = _vmec_print_round(vmec_fsql1, 2)
                    vmec_delt0r = _vmec_print_round(vmec_delt0r, 2)
                    jx_fsqr = _vmec_print_round(jx_fsqr, 2)
                    jx_fsqz = _vmec_print_round(jx_fsqz, 2)
                    jx_fsql = _vmec_print_round(jx_fsql, 2)
                    jx_fsqr1 = _vmec_print_round(jx_fsqr1, 2)
                    jx_fsqz1 = _vmec_print_round(jx_fsqz1, 2)
                    jx_fsql1 = _vmec_print_round(jx_fsql1, 2)
                    jx_delt0r = _vmec_print_round(jx_delt0r, 2)
                    jx_r00 = _vmec_print_round(jx_r00, 3)
                    jx_w = _vmec_print_round(jx_w, 4)
                print(
                    f"  {stage_i+1:>3d} {row.it:>4d}  "
                    f"{vmec_fsqr:>11.3e} {jx_fsqr:>11.3e}  "
                    f"{vmec_fsqz:>11.3e} {jx_fsqz:>11.3e}  "
                    f"{vmec_fsql:>11.3e} {jx_fsql:>11.3e}  "
                    f"{vmec_fsqr1:>11.3e} {jx_fsqr1:>11.3e}  "
                    f"{vmec_fsqz1:>11.3e} {jx_fsqz1:>11.3e}  "
                    f"{vmec_fsql1:>11.3e} {jx_fsql1:>11.3e}  "
                    f"{vmec_delt0r:>11.3e} {jx_delt0r:>11.3e}  "
                    f"{(row.r00 if row.r00 is not None else float('nan')):>11.3e} {jx_r00:>11.3e}  "
                    f"{(row.w if row.w is not None else float('nan')):>11.3e} {jx_w:>11.3e}"
                )
                diff_rows.append((int(stage_i + 1), int(row.it)))
                diff_cols_vmec["fsqr"].append(vmec_fsqr)
                diff_cols_jax["fsqr"].append(jx_fsqr)
                diff_cols_vmec["fsqz"].append(vmec_fsqz)
                diff_cols_jax["fsqz"].append(jx_fsqz)
                diff_cols_vmec["fsql"].append(vmec_fsql)
                diff_cols_jax["fsql"].append(jx_fsql)
                diff_cols_vmec["fsqr1"].append(vmec_fsqr1)
                diff_cols_jax["fsqr1"].append(jx_fsqr1)
                diff_cols_vmec["fsqz1"].append(vmec_fsqz1)
                diff_cols_jax["fsqz1"].append(jx_fsqz1)
                diff_cols_vmec["fsql1"].append(vmec_fsql1)
                diff_cols_jax["fsql1"].append(jx_fsql1)
                diff_cols_vmec["delt0r"].append(vmec_delt0r)
                diff_cols_jax["delt0r"].append(jx_delt0r)
                diff_cols_vmec["r00"].append(float(row.r00 if row.r00 is not None else float("nan")))
                diff_cols_jax["r00"].append(jx_r00)
                diff_cols_vmec["w"].append(float(row.w if row.w is not None else float("nan")))
                diff_cols_jax["w"].append(jx_w)

                if bool(args.fail_fast) and first_mismatch is None:
                    pairs = [
                        ("fsqr", vmec_fsqr, jx_fsqr),
                        ("fsqz", vmec_fsqz, jx_fsqz),
                        ("fsql", vmec_fsql, jx_fsql),
                        ("fsqr1", vmec_fsqr1, jx_fsqr1),
                        ("fsqz1", vmec_fsqz1, jx_fsqz1),
                        ("fsql1", vmec_fsql1, jx_fsql1),
                        ("delt0r", vmec_delt0r, jx_delt0r),
                        ("r00", float(row.r00 if row.r00 is not None else float("nan")), jx_r00),
                        ("wmhd", float(row.w if row.w is not None else float("nan")), jx_w),
                    ]
                    for name, v, jv in pairs:
                        round_step = None
                        if not use_dump_vals:
                            decimals_map = {
                                "fsqr": 2,
                                "fsqz": 2,
                                "fsql": 2,
                                "fsqr1": 2,
                                "fsqz1": 2,
                                "fsql1": 2,
                                "delt0r": 2,
                                "r00": 3,
                                "wmhd": 4,
                            }
                            dec = decimals_map.get(name)
                            if dec is not None:
                                round_step = _round_step(float(v), int(dec))
                        if not _matches(v, jv, field=str(name), round_step=round_step):
                            first_mismatch = {
                                "stage": int(stage_i + 1),
                                "iter": int(row.it),
                                "field": str(name),
                                "vmec": float(v),
                                "jax": float(jv),
                            }
                            stop_after_mismatch = True
                            break
                    if stop_after_mismatch:
                        break
            else:
                assert isinstance(row, Vmec2000PrintedRow)
                print(
                    f"  {stage_i+1:>3d} {row.it:>4d}  "
                    f"{row.fsqr:>11.3e} {fsqr[j]:>11.3e}  "
                    f"{row.fsqz:>11.3e} {fsqz[j]:>11.3e}  "
                    f"{row.fsql:>11.3e} {fsql[j]:>11.3e}"
                )
            if stop_after_mismatch:
                break
        if stop_after_mismatch:
            break

    if first_mismatch is not None:
        print()
        print("First mismatch beyond tolerance:")
        print(
            f"  stage={first_mismatch['stage']} iter={first_mismatch['iter']} field={first_mismatch['field']}"
        )
        print(f"  vmec2000={first_mismatch['vmec']:.6e}  vmec_jax={first_mismatch['jax']:.6e}")
        print(f"  tol: rtol={args.rtol:.3e} atol={args.atol:.3e}")

    if use_threed1 and diff_rows:
        print()
        if vmec_fsq_dump:
            print("Diff summary (max abs / max rel vs VMEC2000 fsq dumps + threed1):")
        else:
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

    if vmec_xc or jax_xc:
        print()
        print("xc/v parity (VMEC2000 dumps vs vmec_jax dumps):")
        vmec_global: dict[int, tuple[int, int, np.ndarray, np.ndarray]] = {}
        jax_global: dict[int, tuple[int, int, np.ndarray, np.ndarray]] = {}
        for (ns_val, it_val), data in vmec_xc.items():
            matches = np.where(vmec_ns == int(ns_val))[0]
            if matches.size == 0:
                continue
            stage_idx = int(matches[0])
            global_it = int(vmec_offsets[stage_idx]) + int(it_val)
            if global_it not in vmec_global:
                vm_xc, vm_v = data
                vmec_global[global_it] = (int(ns_val), int(it_val), vm_xc, vm_v)
        for (ns_val, it_val), data in jax_xc.items():
            matches = np.where(vmec_ns == int(ns_val))[0]
            if matches.size == 0:
                continue
            stage_idx = int(matches[0])
            global_it = int(vmec_offsets[stage_idx]) + int(it_val)
            if global_it not in jax_global:
                jx_xc, jx_v = data
                jax_global[global_it] = (int(ns_val), int(it_val), jx_xc, jx_v)
        common = sorted(set(vmec_global.keys()) & set(jax_global.keys()))
        if not common:
            print("  No overlapping xc dump iterations found.")
        for it in common:
            vm_ns, vm_it, vm_xc, vm_v = vmec_global[it]
            jx_ns, jx_it, jx_xc, jx_v = jax_global[it]
            ns_decode = int(vm_ns if vm_ns is not None else jx_ns)
            keep_xc = _radial_keep_mask_flat(
                size=int(vm_xc.size),
                ns=int(ns_decode),
                radial_skip=int(args.radial_skip),
                radial_drop_edge=bool(args.radial_drop_edge),
            )
            keep_v = _radial_keep_mask_flat(
                size=int(vm_v.size),
                ns=int(ns_decode),
                radial_skip=int(args.radial_skip),
                radial_drop_edge=bool(args.radial_drop_edge),
            )
            vm_xc_cmp = vm_xc[keep_xc] if keep_xc.size == vm_xc.size else vm_xc
            jx_xc_cmp = jx_xc[keep_xc] if keep_xc.size == jx_xc.size else jx_xc
            vm_v_cmp = vm_v[keep_v] if keep_v.size == vm_v.size else vm_v
            jx_v_cmp = jx_v[keep_v] if keep_v.size == jx_v.size else jx_v
            ok_xc, msg_xc, idx_xc = _compare_vectors(
                label="xc",
                vmec_vec=vm_xc_cmp,
                jax_vec=jx_xc_cmp,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
            ok_v, msg_v, idx_v = _compare_vectors(
                label="v",
                vmec_vec=vm_v_cmp,
                jax_vec=jx_v_cmp,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
            print(f"  iter {it:03d} (ns={vm_ns}, vmec_it={vm_it}, jax_it={jx_it}): {msg_xc}; {msg_v}")
            if not ok_xc or not ok_v:
                idx_xc_raw = int(np.flatnonzero(keep_xc)[int(idx_xc)]) if (idx_xc is not None and int(idx_xc) >= 0 and keep_xc.size == vm_xc.size and int(idx_xc) < int(np.count_nonzero(keep_xc))) else int(idx_xc)
                idx_v_raw = int(np.flatnonzero(keep_v)[int(idx_v)]) if (idx_v is not None and int(idx_v) >= 0 and keep_v.size == vm_v.size and int(idx_v) < int(np.count_nonzero(keep_v))) else int(idx_v)
                try:
                    cfg = run.cfg
                    dec_xc = _decode_xc_index(
                        int(idx_xc_raw),
                        ns=int(ns_decode),
                        mpol=int(cfg.mpol),
                        ntor=int(cfg.ntor),
                        lthreed=bool(cfg.lthreed),
                        lasym=bool(getattr(cfg, "lasym", False)),
                    )
                except Exception:
                    dec_xc = "idx decode unavailable"
                try:
                    cfg = run.cfg
                    dec_v = _decode_xc_index(
                        int(idx_v_raw),
                        ns=int(ns_decode),
                        mpol=int(cfg.mpol),
                        ntor=int(cfg.ntor),
                        lthreed=bool(cfg.lthreed),
                        lasym=bool(getattr(cfg, "lasym", False)),
                    )
                except Exception:
                    dec_v = "idx decode unavailable"
                print(f"    xc decode: {dec_xc}")
                print(f"    v decode: {dec_v}")
                if idx_xc is not None and int(idx_xc_raw) >= 0 and int(idx_xc_raw) < vm_xc.size and int(idx_xc_raw) < jx_xc.size:
                    iv = int(idx_xc_raw)
                    v_vm = float(vm_xc[iv])
                    v_jx = float(jx_xc[iv])
                    dv = abs(v_vm - v_jx)
                    rv = dv / max(abs(v_vm), float(args.atol))
                    print(
                        f"    xc values: vmec={v_vm:.16e} jax={v_jx:.16e} abs={dv:.3e} rel={rv:.3e}"
                    )
                if idx_v is not None and int(idx_v_raw) >= 0 and int(idx_v_raw) < vm_v.size and int(idx_v_raw) < jx_v.size:
                    iv = int(idx_v_raw)
                    v_vm = float(vm_v[iv])
                    v_jx = float(jx_v[iv])
                    dv = abs(v_vm - v_jx)
                    rv = dv / max(abs(v_vm), float(args.atol))
                    print(
                        f"    v values:  vmec={v_vm:.16e} jax={v_jx:.16e} abs={dv:.3e} rel={rv:.3e}"
                    )
            if bool(args.fail_fast) and first_mismatch is None and (not ok_xc or not ok_v):
                raise SystemExit(2)

    if "jax_xc_init" in locals() and jax_xc_init:
        print()
        print("xc_init parity (JAX initial guess vs VMEC iter1):")
        cfg = run.cfg
        for (ns_val, label), (jx_xc, jx_v) in sorted(jax_xc_init.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            vm = vmec_xc.get((int(ns_val), 1))
            label_tag = f" {label}" if label else ""
            if vm is None:
                print(f"  ns={int(ns_val)}{label_tag}: VMEC iter1 xc dump not found")
                continue
            vm_xc, vm_v = vm
            ok_xc, msg_xc, idx_xc = _compare_vectors(
                label="xc",
                vmec_vec=vm_xc,
                jax_vec=jx_xc,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
            print(f"  ns={int(ns_val)}{label_tag}: {msg_xc}")
            if not ok_xc:
                try:
                    dec_xc = _decode_xc_index(
                        int(idx_xc),
                        ns=int(ns_val),
                        mpol=int(cfg.mpol),
                        ntor=int(cfg.ntor),
                        lthreed=bool(cfg.lthreed),
                        lasym=bool(getattr(cfg, "lasym", False)),
                    )
                except Exception:
                    dec_xc = "idx decode unavailable"
                print(f"    xc decode: {dec_xc}")
                if idx_xc is not None and int(idx_xc) >= 0 and int(idx_xc) < vm_xc.size and int(idx_xc) < jx_xc.size:
                    iv = int(idx_xc)
                    v_vm = float(vm_xc[iv])
                    v_jx = float(jx_xc[iv])
                    dv = abs(v_vm - v_jx)
                    rv = dv / max(abs(v_vm), float(args.atol))
                    print(
                        f"    xc values: vmec={v_vm:.16e} jax={v_jx:.16e} abs={dv:.3e} rel={rv:.3e}"
                    )
            # xc_init is a diagnostic proxy (JAX init vs VMEC iter1); do not
            # fail-fast on this mismatch.

    if vmec_bsube or jax_bsube:
        print()
        print("bsube parity (VMEC2000 bcovar vs vmec_jax bcovar):")
        common = sorted(vmec_bsube.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping bsube dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx_data = _resolve_other(jax_bsube, ns=ns_val, it=it)
            if jx_data is None:
                continue
            vm_bsubu, vm_bsubv = vmec_bsube[(ns_val, it)]
            jx_bsubu, jx_bsubv = jx_data
            max_abs_u, max_rel_u, _ = _max_abs_rel_err(vm_bsubu.ravel(), jx_bsubu.ravel())
            max_abs_v, max_rel_v, _ = _max_abs_rel_err(vm_bsubv.ravel(), jx_bsubv.ravel())
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            print(
                f"  {ns_tag}iter {it:03d}: bsubu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e};"
                f" bsubv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}"
            )
            if bool(args.fail_fast) and first_mismatch is None:
                tol_u = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubu))))
                tol_v = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubv))))
                if max_abs_u > tol_u or max_abs_v > tol_v:
                    raise SystemExit(2)

    if "vmec_bsubh" in locals() and vmec_bsubh and "jax_bsubh" in locals() and jax_bsubh:
        print()
        print("bsubh parity (VMEC2000 bcovar vs vmec_jax bcovar, |ref|>1e-8):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        common = sorted(vmec_bsubh.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping bsubh dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx_data = _resolve_other(jax_bsubh, ns=ns_val, it=it)
            if jx_data is None:
                continue
            vm_bsubu, vm_bsubv = vmec_bsubh[(ns_val, it)]
            jx_bsubu, jx_bsubv = jx_data
            vm_bsubu = _radial_trim_axis(vm_bsubu, radial_skip, radial_drop_edge, axis=0)
            vm_bsubv = _radial_trim_axis(vm_bsubv, radial_skip, radial_drop_edge, axis=0)
            jx_bsubu = _radial_trim_axis(jx_bsubu, radial_skip, radial_drop_edge, axis=0)
            jx_bsubv = _radial_trim_axis(jx_bsubv, radial_skip, radial_drop_edge, axis=0)
            if vm_bsubu is None or vm_bsubv is None or jx_bsubu is None or jx_bsubv is None:
                continue
            if vm_bsubu.shape != jx_bsubu.shape:
                min_shape = tuple(min(a, b) for a, b in zip(vm_bsubu.shape, jx_bsubu.shape))
                vm_bsubu = vm_bsubu[tuple(slice(0, n) for n in min_shape)]
                jx_bsubu = jx_bsubu[tuple(slice(0, n) for n in min_shape)]
            if vm_bsubv.shape != jx_bsubv.shape:
                min_shape = tuple(min(a, b) for a, b in zip(vm_bsubv.shape, jx_bsubv.shape))
                vm_bsubv = vm_bsubv[tuple(slice(0, n) for n in min_shape)]
                jx_bsubv = jx_bsubv[tuple(slice(0, n) for n in min_shape)]
            mask_u = np.abs(vm_bsubu) > 1e-8
            mask_v = np.abs(vm_bsubv) > 1e-8
            diff_u = np.abs(vm_bsubu - jx_bsubu)
            diff_v = np.abs(vm_bsubv - jx_bsubv)
            if np.any(mask_u):
                max_abs_u = float(np.max(diff_u[mask_u]))
                max_rel_u = float(max_abs_u / max(float(np.max(np.abs(vm_bsubu[mask_u]))), float(args.atol)))
            else:
                max_abs_u = float("nan")
                max_rel_u = float("nan")
            if np.any(mask_v):
                max_abs_v = float(np.max(diff_v[mask_v]))
                max_rel_v = float(max_abs_v / max(float(np.max(np.abs(vm_bsubv[mask_v]))), float(args.atol)))
            else:
                max_abs_v = float("nan")
                max_rel_v = float("nan")
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            print(
                f"  {ns_tag}iter {it:03d}: bsubu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e};"
                f" bsubv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}"
            )
            focus_u = _focus_rel_stats(vm_bsubu, jx_bsubu, focus_abs=1e-3)
            focus_v = _focus_rel_stats(vm_bsubv, jx_bsubv, focus_abs=1e-3)
            if focus_u is not None:
                _fmax_abs_u, fmax_rel_u, frms_rel_u, fn_u, ftotal_u = focus_u
                print(
                    "    bsubu focus |ref|>1e-3:"
                    f" max_rel={fmax_rel_u:.3e} rms_rel={frms_rel_u:.3e}"
                    f" n={fn_u}/{ftotal_u}"
                )
            if focus_v is not None:
                _fmax_abs_v, fmax_rel_v, frms_rel_v, fn_v, ftotal_v = focus_v
                print(
                    "    bsubv focus |ref|>1e-3:"
                    f" max_rel={fmax_rel_v:.3e} rms_rel={frms_rel_v:.3e}"
                    f" n={fn_v}/{ftotal_v}"
                )
            if bool(args.fail_fast) and first_mismatch is None:
                tol_u = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubu[mask_u])))) if np.any(mask_u) else float("inf")
                tol_v = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubv[mask_v])))) if np.any(mask_v) else float("inf")
                if max_abs_u > tol_u or max_abs_v > tol_v:
                    raise SystemExit(2)

    if "vmec_bsub_pre" in locals() and vmec_bsub_pre and "jax_bsub_pre" in locals() and jax_bsub_pre:
        print()
        print("bsub pre-symoutput parity (VMEC2000 wrout vs vmec_jax wout, |ref|>1e-8):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        vm_key = next(iter(vmec_bsub_pre.keys()))
        jx_key = next(iter(jax_bsub_pre.keys()))
        vm_bsubu, vm_bsubv, _, _, _ = vmec_bsub_pre[vm_key]
        jx_bsubu, jx_bsubv, _, _, _ = jax_bsub_pre[jx_key]
        vm_bsubu = _radial_trim_axis(vm_bsubu, radial_skip, radial_drop_edge, axis=0)
        vm_bsubv = _radial_trim_axis(vm_bsubv, radial_skip, radial_drop_edge, axis=0)
        jx_bsubu = _radial_trim_axis(jx_bsubu, radial_skip, radial_drop_edge, axis=0)
        jx_bsubv = _radial_trim_axis(jx_bsubv, radial_skip, radial_drop_edge, axis=0)
        if vm_bsubu is not None and vm_bsubv is not None and jx_bsubu is not None and jx_bsubv is not None:
            min_shape = tuple(min(a, b) for a, b in zip(vm_bsubu.shape, jx_bsubu.shape))
            vm_bsubu = vm_bsubu[tuple(slice(0, n) for n in min_shape)]
            jx_bsubu = jx_bsubu[tuple(slice(0, n) for n in min_shape)]
            min_shape = tuple(min(a, b) for a, b in zip(vm_bsubv.shape, jx_bsubv.shape))
            vm_bsubv = vm_bsubv[tuple(slice(0, n) for n in min_shape)]
            jx_bsubv = jx_bsubv[tuple(slice(0, n) for n in min_shape)]
            mask_u = np.abs(vm_bsubu) > 1e-8
            mask_v = np.abs(vm_bsubv) > 1e-8
            diff_u = np.abs(vm_bsubu - jx_bsubu)
            diff_v = np.abs(vm_bsubv - jx_bsubv)
            max_abs_u = float(np.max(diff_u[mask_u])) if np.any(mask_u) else float("nan")
            max_abs_v = float(np.max(diff_v[mask_v])) if np.any(mask_v) else float("nan")
            max_rel_u = (
                float(max_abs_u / max(float(np.max(np.abs(vm_bsubu[mask_u]))), float(args.atol)))
                if np.any(mask_u)
                else float("nan")
            )
            max_rel_v = (
                float(max_abs_v / max(float(np.max(np.abs(vm_bsubv[mask_v]))), float(args.atol)))
                if np.any(mask_v)
                else float("nan")
            )
            print(
                f"  bsubu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e};"
                f" bsubv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}"
            )
            focus_u = _focus_rel_stats(vm_bsubu, jx_bsubu, focus_abs=1e-3)
            focus_v = _focus_rel_stats(vm_bsubv, jx_bsubv, focus_abs=1e-3)
            if focus_u is not None:
                _fmax_abs_u, fmax_rel_u, frms_rel_u, fn_u, ftotal_u = focus_u
                print(
                    "  bsubu focus |ref|>1e-3:"
                    f" max_rel={fmax_rel_u:.3e} rms_rel={frms_rel_u:.3e}"
                    f" n={fn_u}/{ftotal_u}"
                )
            if focus_v is not None:
                _fmax_abs_v, fmax_rel_v, frms_rel_v, fn_v, ftotal_v = focus_v
                print(
                    "  bsubv focus |ref|>1e-3:"
                    f" max_rel={fmax_rel_v:.3e} rms_rel={frms_rel_v:.3e}"
                    f" n={fn_v}/{ftotal_v}"
                )

    if "vmec_bsube_terms" in locals() and (vmec_bsube_terms or jax_bsube_terms):
        print()
        print("bsube terms parity (lvv_sh/lu0/lu1/phipf/bsubu_tmp/bsubv_pre):")
        common = sorted(vmec_bsube_terms.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping bsube_terms dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx_data = _resolve_other(jax_bsube_terms, ns=ns_val, it=it)
            if jx_data is None:
                continue
            vm = vmec_bsube_terms[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("lvv_sh", "lu0", "lu1", "phipf", "bsubu_tmp", "bsubv_pre"):
                vm_arr = np.asarray(vm.get(name))
                jx_arr = np.asarray(jx_data.get(name))
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                flat_vm = vm_arr.ravel()
                flat_jx = jx_arr.ravel()
                diff = np.abs(flat_vm - flat_jx)
                idx = int(np.argmax(diff)) if diff.size else 0
                max_abs = float(diff[idx]) if diff.size else float("nan")
                max_rel = float(max_abs / max(np.abs(flat_vm[idx]), float(args.atol))) if diff.size else float("nan")
                if vm_arr.ndim == 3:
                    js_i, lt_i, lz_i = np.unravel_index(idx, vm_arr.shape)
                    decode = f" idx=js={js_i+1} lt={lt_i+1} lz={lz_i+1}"
                else:
                    decode = f" idx={idx}"
                print(
                    f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}{decode}"
                )
                if diff.size:
                    v = float(flat_vm[idx])
                    j = float(flat_jx[idx])
                    print(f"    {name} values: vmec={v:.6e} jax={j:.6e}")
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_arr))))
                    if max_abs > tol:
                        raise SystemExit(2)

    if "vmec_bsubs" in locals() and vmec_bsubs and "jax_bsubs" in locals() and jax_bsubs:
        print()
        print("bsubs parity (VMEC2000 bsubs dump vs vmec_jax bsubs):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        common = sorted(vmec_bsubs.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping bsubs dump iterations found.")
        for ns_val, it in common:
            vm_bsubs = np.asarray(vmec_bsubs[(ns_val, it)])
            candidates: list[tuple[float, float, int | None, int, str]] = []
            for it_jx in (it, it - 1, it + 1):
                if it_jx <= 0:
                    continue
                ns_jx, jx_data = _resolve_other(jax_bsubs, ns=ns_val, it=it_jx)
                if jx_data is None:
                    continue
                for jx_key in ("bsubs_half", "bsubs_full"):
                    jx_raw = jx_data.get(jx_key)
                    if jx_raw is None:
                        continue
                    vm_cmp = _radial_trim_axis(vm_bsubs, radial_skip, radial_drop_edge, axis=0)
                    jx_cmp = _radial_trim_axis(np.asarray(jx_raw), radial_skip, radial_drop_edge, axis=0)
                    if vm_cmp is None or jx_cmp is None:
                        continue
                    if vm_cmp.shape != jx_cmp.shape:
                        min_shape = tuple(min(a, b) for a, b in zip(vm_cmp.shape, jx_cmp.shape))
                        vm_cmp = vm_cmp[tuple(slice(0, n) for n in min_shape)]
                        jx_cmp = jx_cmp[tuple(slice(0, n) for n in min_shape)]
                    max_abs, max_rel, _ = _max_abs_rel_err(vm_cmp.ravel(), jx_cmp.ravel())
                    iter_note = "" if it_jx == it else f" (jax iter {it_jx})"
                    iter_note = iter_note + f" {jx_key}"
                    candidates.append((max_abs, max_rel, ns_jx, it_jx, iter_note))
            if not candidates:
                continue
            max_abs, max_rel, ns_jx, _it_jx, iter_note = min(candidates, key=lambda x: x[0])
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            print(f"  {ns_tag}iter {it:03d}{iter_note}: bsubs max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_cmp))))
                if max_abs > tol:
                    raise SystemExit(2)

    if "vmec_jxbout" in locals() and vmec_jxbout and "jax_bsubs" in locals() and jax_bsubs:
        print()
        print("bsubs parity (VMEC2000 jxbout vs vmec_jax bsubs):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        key = max(jax_bsubs.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        jx = jax_bsubs[key]
        bsubs_full = jx.get("bsubs_full")
        if bsubs_full is None:
            bsubs_full = jx.get("bsubs_half")
        bsubs_full = np.asarray(bsubs_full)
        # jxbout stores arrays as (ntheta, nzeta, ns)
        bsubs_jx = np.transpose(bsubs_full, (1, 2, 0))
        vm_bsubs = vmec_jxbout.get("bsubs")
        if vm_bsubs is None:
            print("  VMEC jxbout missing bsubs array.")
        else:
            vm_bsubs = _radial_trim_axis(vm_bsubs, radial_skip, radial_drop_edge, axis=2)
            bsubs_jx = _radial_trim_axis(bsubs_jx, radial_skip, radial_drop_edge, axis=2)
        if vm_bsubs is None or bsubs_jx is None:
            print("  bsubs parity skipped (no bsubs data after trim).")
        elif vm_bsubs.shape != bsubs_jx.shape:
            print(f"  Shape mismatch: vmec={vm_bsubs.shape} jax={bsubs_jx.shape}")
        else:
            max_abs, max_rel, _ = _max_abs_rel_err(vm_bsubs.ravel(), bsubs_jx.ravel())
            print(f"  bsubs max_abs={max_abs:.3e} max_rel={max_rel:.3e} (iter={key[1]})")
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubs))))
                if max_abs > tol:
                    raise SystemExit(2)

        vm_bsupu = vmec_jxbout.get("bsupu")
        vm_bsupv = vmec_jxbout.get("bsupv")
        if vm_bsupu is not None and vm_bsupv is not None:
            bsupu1 = np.asarray(jx.get("bsupu1"))
            bsupv1 = np.asarray(jx.get("bsupv1"))
            bsupu_jx = np.transpose(bsupu1, (1, 2, 0))
            bsupv_jx = np.transpose(bsupv1, (1, 2, 0))
            vm_bsupu = _radial_trim_axis(vm_bsupu, radial_skip, radial_drop_edge, axis=2)
            vm_bsupv = _radial_trim_axis(vm_bsupv, radial_skip, radial_drop_edge, axis=2)
            bsupu_jx = _radial_trim_axis(bsupu_jx, radial_skip, radial_drop_edge, axis=2)
            bsupv_jx = _radial_trim_axis(bsupv_jx, radial_skip, radial_drop_edge, axis=2)
            if (
                vm_bsupu is not None
                and bsupu_jx is not None
                and vm_bsupv is not None
                and bsupv_jx is not None
                and vm_bsupu.shape == bsupu_jx.shape
            ):
                max_abs_u, max_rel_u, _ = _max_abs_rel_err(vm_bsupu.ravel(), bsupu_jx.ravel())
                max_abs_v, max_rel_v, _ = _max_abs_rel_err(vm_bsupv.ravel(), bsupv_jx.ravel())
                print(f"  bsupu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e}")
                print(f"  bsupv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}")

        vm_bsubu = vmec_jxbout.get("bsubu")
        vm_bsubv = vmec_jxbout.get("bsubv")
        if vm_bsubu is not None and vm_bsubv is not None:
            bsubu_jx = np.asarray(jx.get("bsubu_jxbout", jx.get("bsubu")))
            bsubv_jx = np.asarray(jx.get("bsubv_jxbout", jx.get("bsubv")))
            if bsubu_jx is not None and bsubv_jx is not None:
                bsubu_jx = np.transpose(bsubu_jx, (1, 2, 0))
                bsubv_jx = np.transpose(bsubv_jx, (1, 2, 0))
                vm_bsubu = _radial_trim_axis(vm_bsubu, radial_skip, radial_drop_edge, axis=2)
                vm_bsubv = _radial_trim_axis(vm_bsubv, radial_skip, radial_drop_edge, axis=2)
                bsubu_jx = _radial_trim_axis(bsubu_jx, radial_skip, radial_drop_edge, axis=2)
                bsubv_jx = _radial_trim_axis(bsubv_jx, radial_skip, radial_drop_edge, axis=2)
                if (
                    vm_bsubu is not None
                    and bsubu_jx is not None
                    and vm_bsubv is not None
                    and bsubv_jx is not None
                    and vm_bsubu.shape == bsubu_jx.shape
                ):
                    max_abs_u, max_rel_u, _ = _max_abs_rel_err(vm_bsubu.ravel(), bsubu_jx.ravel())
                    max_abs_v, max_rel_v, _ = _max_abs_rel_err(vm_bsubv.ravel(), bsubv_jx.ravel())
                    print(f"  bsubu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e}")
                    print(f"  bsubv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}")
                else:
                    print(f"  bsubu/bsubv shape mismatch: vmec={vm_bsubu.shape} jax={bsubu_jx.shape}")

        vm_bdotk = vmec_jxbout.get("sqrt_g__bdotk")
        if vm_bdotk is not None:
            # Compare to bdotk reconstructed from jxbforce parity dump if present.
            # Our bdotk is stored in the jdotb debug npz; skip if not available.
            pass

    if "vmec_bss_inputs" in locals() and vmec_bss_inputs and "jax_bss_inputs" in locals() and jax_bss_inputs:
        print()
        print("bss inputs parity (VMEC2000 vs vmec_jax):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        for it, vm in sorted(vmec_bss_inputs.items()):
            print(f"  iter {it:03d}:")
            for name in ("r12", "rs", "zs", "ru12", "zu12", "bsupu", "bsupv"):
                vm_arr = _radial_trim_axis(vm.get(name), radial_skip, radial_drop_edge, axis=0)
                jx_arr = _radial_trim_axis(jax_bss_inputs.get(name), radial_skip, radial_drop_edge, axis=0)
                if vm_arr is None or jx_arr is None:
                    continue
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                max_abs, max_rel, _ = _max_abs_rel_err(vm_arr.ravel(), jx_arr.ravel())
                print(f"    {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    if "vmec_bss_terms" in locals() and vmec_bss_terms and "jax_bss_terms" in locals() and jax_bss_terms:
        print()
        print("bss terms parity (VMEC2000 vs vmec_jax):")
        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)
        for it, vm in sorted(vmec_bss_terms.items()):
            print(f"  iter {it:03d}:")
            for name in (
                "r1_even",
                "r1_odd",
                "pr1_even",
                "pr1_odd",
                "rv_even",
                "rv_odd",
                "prv_even",
                "prv_odd",
                "ru_even",
                "ru_odd",
                "zu_even",
                "zu_odd",
            ):
                vm_arr = _radial_trim_axis(vm.get(name), radial_skip, radial_drop_edge, axis=0)
                jx_arr = _radial_trim_axis(jax_bss_terms.get(name), radial_skip, radial_drop_edge, axis=0)
                if vm_arr is None or jx_arr is None:
                    continue
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                max_abs, max_rel, _ = _max_abs_rel_err(vm_arr.ravel(), jx_arr.ravel())
                print(f"    {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

    if "vmec_jacobian_terms" in locals() and vmec_jacobian_terms and "jax_kernels" in locals() and jax_kernels:
        print()
        print("jacobian parity (even/odd inputs + ru12/zs/zu12/rs/r12/tau):")
        common = sorted(vmec_jacobian_terms.keys())
        if not common:
            print("  No jacobian_terms dump iterations found.")
        for it in common:
            vm = vmec_jacobian_terms[it]
            ns_vm = int(vm["ru12"].shape[0]) if vm.get("ru12") is not None else None
            ns_jx, jx_data = _resolve_other(jax_kernels, ns=ns_vm, it=it)
            if jx_data is None:
                continue
            ns_print = ns_vm if ns_vm is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name, jx_name in (
                ("pru_e", "pru_even"),
                ("pru_o", "pru_odd"),
                ("pz1_e", "pz1_even"),
                ("pz1_o", "pz1_odd"),
                ("pzu_e", "pzu_even"),
                ("pzu_o", "pzu_odd"),
                ("pr1_e", "pr1_even"),
                ("pr1_o", "pr1_odd"),
                ("prv_e", "prv_even"),
                ("prv_o", "prv_odd"),
                ("pzv_e", "pzv_even"),
                ("pzv_o", "pzv_odd"),
            ):
                vm_arr = np.asarray(vm.get(name))
                jx_arr = np.asarray(jx_data.get(jx_name))
                if vm_arr.shape[0] > 1:
                    vm_arr = vm_arr[1:]
                    jx_arr = jx_arr[1:]
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                max_abs, max_rel, idx = _max_abs_rel_err(vm_arr.ravel(), jx_arr.ravel())
                if vm_arr.ndim == 3:
                    js_i, lt_i, lz_i = np.unravel_index(idx, vm_arr.shape)
                    decode = f" idx=js={js_i+1} lt={lt_i+1} lz={lz_i+1}"
                else:
                    decode = f" idx={idx}"
                print(
                    f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}{decode}"
                )
            for name in ("ru12", "zs", "zu12", "rs", "r12", "tau"):
                vm_arr = np.asarray(vm.get(name))
                jx_arr = np.asarray(jx_data.get(name))
                if vm_arr.shape[0] > 1:
                    vm_arr = vm_arr[1:]
                    jx_arr = jx_arr[1:]
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                max_abs, max_rel, idx = _max_abs_rel_err(vm_arr.ravel(), jx_arr.ravel())
                if vm_arr.ndim == 3:
                    js_i, lt_i, lz_i = np.unravel_index(idx, vm_arr.shape)
                    decode = f" idx=js={js_i+1} lt={lt_i+1} lz={lz_i+1}"
                else:
                    decode = f" idx={idx}"
                print(
                    f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}{decode}"
                )
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_arr))))
                    if max_abs > tol:
                        raise SystemExit(2)

    if "vmec_lulv" in locals() and (vmec_lulv or jax_lulv):
        print()
        print("lu/lv parity (real-space LU/LV even/odd):")
        common = sorted(vmec_lulv.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping lulv dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx_data = _resolve_other(jax_lulv, ns=ns_val, it=it)
            if jx_data is None:
                continue
            vm = vmec_lulv[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("lu0", "lu1", "lv0", "lv1"):
                vm_arr = np.asarray(vm.get(name))
                jx_arr = np.asarray(jx_data.get(name))
                if vm_arr.shape != jx_arr.shape:
                    min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                    vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                    jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
                flat_vm = vm_arr.ravel()
                flat_jx = jx_arr.ravel()
                diff = np.abs(flat_vm - flat_jx)
                idx = int(np.argmax(diff)) if diff.size else 0
                max_abs = float(diff[idx]) if diff.size else float("nan")
                max_rel = float(max_abs / max(np.abs(flat_vm[idx]), float(args.atol))) if diff.size else float("nan")
                if vm_arr.ndim == 3:
                    js_i, lt_i, lz_i = np.unravel_index(idx, vm_arr.shape)
                    decode = f" idx=js={js_i+1} lt={lt_i+1} lz={lz_i+1}"
                else:
                    decode = f" idx={idx}"
                print(
                    f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}{decode}"
                )
                if diff.size:
                    v = float(flat_vm[idx])
                    j = float(flat_jx[idx])
                    print(f"    {name} values: vmec={v:.6e} jax={j:.6e}")
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_arr))))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_scalars or jax_scalars:
        print()
        print("bcovar scalar parity (wb/wp/volume/r2/fnorm/fnormL):")
        common = sorted(vmec_scalars.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping scalars dump iterations found.")
        for ns_val, it in common:
            if ns_val is not None:
                jx = jax_scalars.get((ns_val, it))
                ns_jx = ns_val if jx is not None else None
            else:
                ns_jx, jx = _resolve_other(jax_scalars, ns=ns_val, it=it)
            if jx is None:
                continue
            vm = vmec_scalars[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("wb", "wp", "volume", "r2", "fnorm", "fnormL"):
                v = float(vm.get(name, float("nan")))
                j = float(jx.get(name, float("nan")))
                max_abs = abs(v - j)
                max_rel = max_abs / max(abs(v), float(args.atol)) if np.isfinite(v) else float("nan")
                print(f"  {ns_tag}iter {it:03d} {name}: vmec={v:.6e} jax={j:.6e} abs={max_abs:.3e} rel={max_rel:.3e}")
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * abs(v))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_gcx2 or jax_gcx2:
        print()
        print("gcx2 parity (post-scalxc/m1 sums):")
        common = sorted(vmec_gcx2.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping gcx2 dump iterations found.")
        for ns_val, it in common:
            if ns_val is not None:
                jx = jax_gcx2.get((ns_val, it))
                ns_jx = ns_val if jx is not None else None
            else:
                ns_jx, jx = _resolve_other(jax_gcx2, ns=ns_val, it=it)
            if jx is None:
                continue
            vm = vmec_gcx2[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("gcr2", "gcz2", "gcl2"):
                v = float(vm.get(name, float("nan")))
                j = float(jx.get(name, float("nan")))
                max_abs = abs(v - j)
                max_rel = max_abs / max(abs(v), float(args.atol)) if np.isfinite(v) else float("nan")
                print(f"  {ns_tag}iter {it:03d} {name}: vmec={v:.6e} jax={j:.6e} abs={max_abs:.3e} rel={max_rel:.3e}")
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * abs(v))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_gcx2 and (include_edge_hist.size or zero_m1_hist.size):
        print()
        print("gating parity (include_edge / zero_m1):")
        for ns_val, it in sorted(vmec_gcx2.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1])):
            if ns_val is None:
                continue
            matches = np.where(vmec_ns == int(ns_val))[0]
            if matches.size == 0:
                continue
            stage_idx = int(matches[0])
            idx = int(vmec_offsets[stage_idx]) + max(int(it) - 1, 0)
            if include_edge_hist.size:
                vm_edge = int(round(float(vmec_gcx2[(ns_val, it)].get("include_edge", 0.0))))
                jx_edge = int(include_edge_hist[idx]) if idx < include_edge_hist.size else -1
                print(f"  ns={ns_val} iter {it:03d} include_edge: vmec={vm_edge} jax={jx_edge}")
                if bool(args.fail_fast) and first_mismatch is None and (jx_edge >= 0) and (vm_edge != jx_edge):
                    raise SystemExit(2)
            if zero_m1_hist.size and vmec_fsq_dump:
                if it < 2:
                    vm_zero_m1 = 1
                else:
                    prev = _lookup_by_ns(vmec_fsq_dump, ns=int(ns_val), it=int(it) - 1) or {}
                    fsqz_prev = float(prev.get("fsqz", 0.0))
                    vm_zero_m1 = 1 if fsqz_prev < 1.0e-6 else 0
                jx_zero_m1 = int(zero_m1_hist[idx]) if idx < zero_m1_hist.size else -1
                print(f"  ns={ns_val} iter {it:03d} zero_m1: vmec={vm_zero_m1} jax={jx_zero_m1}")
                if bool(args.fail_fast) and first_mismatch is None and (jx_zero_m1 >= 0) and (vm_zero_m1 != jx_zero_m1):
                    raise SystemExit(2)

    if vmec_kernels or jax_kernels:
        print()
        print("tomnsps kernels parity (VMEC2000 vs vmec_jax force kernels):")
        common = sorted(vmec_kernels.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping tomnsps_kernels/force_kernels dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx = _resolve_other(jax_kernels, ns=ns_val, it=it)
            if jx is None:
                continue
            vm = vmec_kernels[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("blmn", "clmn"):
                v = np.asarray(vm[name]).ravel()
                j = np.asarray(jx.get(name, np.zeros_like(vm[name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(v, j)
                msg = f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                if idx >= 0:
                    msg += f" idx={_format_kernel_index(int(idx), shape=vm[name].shape)}"
                print(msg)
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(v))))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_tomnsps or jax_tomnsps:
        print()
        print("tomnsps parity (VMEC2000 vs vmec_jax raw blocks):")
        common = sorted(vmec_tomnsps.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping tomnsps dump iterations found.")
        for ns_val, it in common:
            ns_jx, jx = _resolve_other(jax_tomnsps, ns=ns_val, it=it)
            if jx is None:
                continue
            vm = vmec_tomnsps[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for name in ("frcc", "frss", "fzsc", "fzcs", "flsc", "flcs"):
                v = np.asarray(vm[name]).ravel()
                j = np.asarray(jx.get(name, np.zeros_like(vm[name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(v, j)
                print(f"  {ns_tag}iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
                if bool(args.fail_fast) and first_mismatch is None:
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(v))))
                    if max_abs > tol:
                        if name == "flsc":
                            decode = _decode_tomnsps_index(idx, vm[name].shape)
                            print(f"    flsc mismatch idx={decode}")
                        raise SystemExit(2)

    # Tighten lambda raw blocks: normalize against RMS and fail-fast with decoded indices.
    if vmec_tomnsps and jax_tomnsps and vmec_gc and jax_gc:
        print()
        print("lambda raw-block normalization (flsc/gcl):")
        common = sorted(vmec_tomnsps.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        raw_gcl_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
        raw_gcl_jx = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "raw"}
        common_gcl = sorted(raw_gcl_vm.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        for ns_val, it in common:
            ns_jx, jx_tomn = _resolve_other(jax_tomnsps, ns=ns_val, it=it)
            if jx_tomn is None:
                continue
            vm_fl = np.asarray(vmec_tomnsps[(ns_val, it)]["flsc"])
            jx_fl = np.asarray(jx_tomn.get("flsc", np.zeros_like(vm_fl)))
            max_abs, _max_rel, idx = _max_abs_rel_err(vm_fl.ravel(), jx_fl.ravel())
            denom = _rms(vm_fl)
            norm_err = max_abs / denom if denom > 0 else float("nan")
            decode = _decode_tomnsps_index(idx, vm_fl.shape)
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            print(f"  {ns_tag}iter {it:03d} flsc: rms={denom:.3e} max_abs={max_abs:.3e} norm_err={norm_err:.3e} idx={decode}")
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * denom)
                if max_abs > tol:
                    raise SystemExit(2)
        for ns_val, it in common_gcl:
            ns_jx, jx_gcl = _resolve_other(raw_gcl_jx, ns=ns_val, it=it)
            if jx_gcl is None:
                continue
            vm_gcl = np.asarray(raw_gcl_vm[(ns_val, it)])
            max_abs, _max_rel, idx = _max_abs_rel_err(vm_gcl.ravel(), jx_gcl.ravel())
            denom = _rms(vm_gcl)
            norm_err = max_abs / denom if denom > 0 else float("nan")
            decode = _decode_gc_index(idx, vm_gcl.shape)
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            print(f"  {ns_tag}iter {it:03d} gcl(raw): rms={denom:.3e} max_abs={max_abs:.3e} norm_err={norm_err:.3e} idx={decode}")
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * denom)
                if max_abs > tol:
                    raise SystemExit(2)

    if (vmec_lam or jax_lam) and vmec_gc and jax_gc:
        print()
        print("lambda preconditioner parity (pfaclam/faclam):")
        common = sorted(vmec_lam.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping lam dumps found.")
        for ns_val, it in common:
            ns_jx, jx_vals = _resolve_other(jax_lam, ns=ns_val, it=it)
            if jx_vals is None:
                continue
            vm_vals = vmec_lam[(ns_val, it)]
            pf_vm = np.asarray(vm_vals.get("pfaclam", np.zeros((0,), dtype=float)))
            pf_jx = np.asarray(jx_vals.get("pfaclam", np.zeros((0,), dtype=float)))
            pf_vm, pf_jx = _align_lam_arrays(pf_vm, pf_jx)
            max_abs, max_rel, idx = _max_abs_rel_err(pf_vm.ravel(), pf_jx.ravel())
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            msg = f"  {ns_tag}iter {it:03d} pfaclam: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
            if idx >= 0 and pf_vm.ndim == 4:
                msg += f" idx={_decode_gc_index(int(idx), pf_vm.shape)}"
            print(msg)
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(pf_vm))))
                if max_abs > tol:
                    raise SystemExit(2)
            fac_vm = vm_vals.get("faclam", None)
            fac_jx = jx_vals.get("faclam", None)
            if fac_vm is not None and fac_jx is not None:
                fac_vm, fac_jx = _align_lam_arrays(np.asarray(fac_vm), np.asarray(fac_jx))
                max_abs, max_rel, idx = _max_abs_rel_err(fac_vm.ravel(), fac_jx.ravel())
                msg = f"  {ns_tag}iter {it:03d} faclam:  max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                if idx >= 0 and fac_vm.ndim == 4:
                    msg += f" idx={_decode_gc_index(int(idx), fac_vm.shape)}"
                print(msg)

    if "vmec_lamcal" in locals() and (vmec_lamcal or jax_lamcal):
        print()
        print("lamcal parity (blam/clam/dlam pre/post average):")
        common = sorted(vmec_lamcal.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        if not common:
            print("  No overlapping lamcal dumps found.")
        for ns_val, it in common:
            ns_jx, jx_vals = _resolve_other(jax_lamcal, ns=ns_val, it=it)
            if jx_vals is None:
                continue
            vm_vals = vmec_lamcal[(ns_val, it)]
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            for stage_name, vm_key, jx_key in (
                ("pre", "pre", ("blam_pre", "clam_pre", "dlam_pre")),
                ("post", "post", ("blam_post", "clam_post", "dlam_post")),
            ):
                if vm_key not in vm_vals:
                    continue
                vm_stage = vm_vals[vm_key]
                for comp, jx_name in zip(("blam", "clam", "dlam"), jx_key):
                    vm_arr = np.asarray(vm_stage.get(comp))
                    jx_arr = np.asarray(jx_vals.get(jx_name))
                    if vm_arr.shape != jx_arr.shape:
                        n = min(vm_arr.shape[0], jx_arr.shape[0])
                        vm_arr = vm_arr[:n]
                        jx_arr = jx_arr[:n]
                    diff = np.abs(vm_arr - jx_arr)
                    idx = int(np.argmax(diff)) if diff.size else 0
                    max_abs = float(diff[idx]) if diff.size else float("nan")
                    max_rel = float(max_abs / max(np.abs(vm_arr[idx]), float(args.atol))) if diff.size else float("nan")
                    print(
                        f"  {ns_tag}iter {it:03d} {stage_name} {comp}: max_abs={max_abs:.3e} "
                        f"max_rel={max_rel:.3e} idx={idx+1}"
                    )
                    if diff.size:
                        v = float(vm_arr[idx])
                        j = float(jx_arr[idx])
                        print(f"    {stage_name} {comp} values: vmec={v:.6e} jax={j:.6e}")
                    if bool(args.fail_fast) and first_mismatch is None:
                        tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_arr))))
                        if max_abs > tol:
                            raise SystemExit(2)

    if vmec_gc and jax_gc:
        print()
        print("lambda fsql1 components (pre/post faclam):")
        raw_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
        pre_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "precond"}
        raw_jx = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "raw"}
        pre_jx = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "precond"}
        common = sorted(raw_vm.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        for ns_val, it in common:
            if (ns_val, it) not in pre_vm:
                continue
            ns_jx, jx_raw = _resolve_other(raw_jx, ns=ns_val, it=it)
            _, jx_pre = _resolve_other(pre_jx, ns=ns_val, it=it)
            if jx_raw is None or jx_pre is None:
                continue
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_use = int(ns_print) if ns_print is not None else int(np.asarray(jx_raw).shape[0])
            delta_s = 1.0 / max(ns_use - 1, 1)
            vm_pre = vmec_lam_fsql1.get((ns_val, it)) if "vmec_lam_fsql1" in locals() else None
            jx_pre = jax_lam_fsql1.get((ns_val, it)) if "jax_lam_fsql1" in locals() else None
            vm_preval = float(vm_pre["fsql1_pre"]) if vm_pre is not None else _fsql1_from_gcl(raw_vm[(ns_val, it)], delta_s)
            vm_postval = float(vm_pre["fsql1_post"]) if vm_pre is not None else _fsql1_from_gcl(pre_vm[(ns_val, it)], delta_s)
            jx_preval = float(jx_pre["fsql1_pre"]) if jx_pre is not None else _fsql1_from_gcl(jx_raw, delta_s)
            jx_postval = float(jx_pre["fsql1_post"]) if jx_pre is not None else _fsql1_from_gcl(jx_pre, delta_s)
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            pre_abs = abs(vm_preval - jx_preval)
            post_abs = abs(vm_postval - jx_postval)
            pre_rel = pre_abs / max(abs(vm_preval), float(args.atol), 1e-30)
            post_rel = post_abs / max(abs(vm_postval), float(args.atol), 1e-30)
            print(
                f"  {ns_tag}iter {it:03d} fsql1_pre vmec={vm_preval:.3e} jax={jx_preval:.3e} rel={pre_rel:.3e} |"
                f" fsql1_post vmec={vm_postval:.3e} jax={jx_postval:.3e} rel={post_rel:.3e}"
            )
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * float(abs(vm_postval)))
                if post_abs > tol:
                    raise SystemExit(2)

    if "jax_lam_gcl" in locals() and jax_lam_gcl and vmec_gc:
        print()
        print("lambda gcl per-mode (pre/post faclam):")
        raw_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
        pre_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "precond"}
        for (ns_val, it), jx_vals in sorted(jax_lam_gcl.items(), key=lambda k: (k[0][0] is None, k[0][0] or -1, k[0][1])):
            vm_raw = raw_vm.get((ns_val, it))
            vm_pre = pre_vm.get((ns_val, it))
            if vm_raw is None or vm_pre is None:
                continue
            gcl_pre = np.asarray(jx_vals.get("gcl_pre", np.zeros_like(vm_raw)))
            gcl_post = np.asarray(jx_vals.get("gcl_post", np.zeros_like(vm_pre)))
            vm_raw = np.asarray(vm_raw)
            vm_pre = np.asarray(vm_pre)
            if gcl_pre.ndim == 4 and vm_raw.ndim == 4:
                if gcl_pre.shape[1] == vm_raw.shape[2] and gcl_pre.shape[2] == vm_raw.shape[1]:
                    gcl_pre = np.transpose(gcl_pre, (0, 2, 1, 3))
            if gcl_post.ndim == 4 and vm_pre.ndim == 4:
                if gcl_post.shape[1] == vm_pre.shape[2] and gcl_post.shape[2] == vm_pre.shape[1]:
                    gcl_post = np.transpose(gcl_post, (0, 2, 1, 3))
            if gcl_pre.shape != vm_raw.shape:
                min_shape = tuple(min(a, b) for a, b in zip(gcl_pre.shape, vm_raw.shape))
                gcl_pre = gcl_pre[tuple(slice(0, n) for n in min_shape)]
                vm_raw = vm_raw[tuple(slice(0, n) for n in min_shape)]
            if gcl_post.shape != vm_pre.shape:
                min_shape = tuple(min(a, b) for a, b in zip(gcl_post.shape, vm_pre.shape))
                gcl_post = gcl_post[tuple(slice(0, n) for n in min_shape)]
                vm_pre = vm_pre[tuple(slice(0, n) for n in min_shape)]
            pre_abs, pre_rel, pre_idx = _max_abs_rel_err(vm_raw.ravel(), gcl_pre.ravel())
            post_abs, post_rel, post_idx = _max_abs_rel_err(vm_pre.ravel(), gcl_post.ravel())
            ns_tag = f"ns={ns_val} " if ns_val is not None else ""
            msg = (
                f"  {ns_tag}iter {it:03d} gcl_pre: max_abs={pre_abs:.3e} max_rel={pre_rel:.3e}"
                f" | gcl_post: max_abs={post_abs:.3e} max_rel={post_rel:.3e}"
            )
            if pre_idx >= 0 and vm_raw.ndim == 4:
                msg += f" pre_idx={_decode_gc_index(int(pre_idx), vm_raw.shape)}"
            if post_idx >= 0 and vm_pre.ndim == 4:
                msg += f" post_idx={_decode_gc_index(int(post_idx), vm_pre.shape)}"
            print(msg)
            if bool(args.fail_fast) and first_mismatch is None:
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_pre))))
                if post_abs > tol:
                    raise SystemExit(2)

    if (not vmec_lam or not jax_lam) and vmec_gc and jax_gc:
        print()
        print("lambda effective pfaclam ratio (gcl_pre/gcl_raw):")
        raw_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
        pre_vm = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "precond"}
        raw_jx = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "raw"}
        pre_jx = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "precond"}
        common = sorted(raw_vm.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
        for ns_val, it in common:
            if (ns_val, it) not in pre_vm:
                continue
            ns_jx, jx_raw = _resolve_other(raw_jx, ns=ns_val, it=it)
            _, jx_pre = _resolve_other(pre_jx, ns=ns_val, it=it)
            if jx_raw is None or jx_pre is None:
                continue
            vm_raw = np.asarray(raw_vm[(ns_val, it)], dtype=float)
            vm_pre = np.asarray(pre_vm[(ns_val, it)], dtype=float)
            jx_raw = np.asarray(jx_raw, dtype=float)
            jx_pre = np.asarray(jx_pre, dtype=float)
            eps_vm = max(1e-30, 1e-12 * float(np.nanmax(np.abs(vm_raw))))
            eps_jx = max(1e-30, 1e-12 * float(np.nanmax(np.abs(jx_raw))))
            mask_vm = np.abs(vm_raw) > eps_vm
            mask_jx = np.abs(jx_raw) > eps_jx
            pf_vm = np.zeros_like(vm_raw)
            pf_jx = np.zeros_like(jx_raw)
            pf_vm[mask_vm] = vm_pre[mask_vm] / vm_raw[mask_vm]
            pf_jx[mask_jx] = jx_pre[mask_jx] / jx_raw[mask_jx]
            max_abs, max_rel, idx = _max_abs_rel_err(pf_vm.ravel(), pf_jx.ravel())
            ns_print = ns_val if ns_val is not None else ns_jx
            ns_tag = f"ns={ns_print} " if ns_print is not None else ""
            msg = f"  {ns_tag}iter {it:03d} pfaclam_eff: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
            if idx >= 0 and pf_vm.ndim == 4:
                msg += f" idx={_decode_gc_index(int(idx), pf_vm.shape)}"
            print(msg)

    # Lambda-path audit: flsc/gcl vs blmn/clmn at the first mismatch index.
    if (vmec_tomnsps and jax_tomnsps) or (vmec_gc and jax_gc) or (vmec_kernels and jax_kernels):
        print()
        print("lambda-path audit (first mismatch across flsc/gcl/blmn/clmn):")

        def _first_block_mismatch(block_name: str) -> tuple[int | None, int, int, float, float] | None:
            common = sorted(vmec_tomnsps.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
            for ns_val, it in common:
                ns_jx, jx_tomn = _resolve_other(jax_tomnsps, ns=ns_val, it=it)
                if jx_tomn is None:
                    continue
                vm = np.asarray(vmec_tomnsps[(ns_val, it)][block_name]).ravel()
                jx = np.asarray(jx_tomn.get(block_name, np.zeros_like(vmec_tomnsps[(ns_val, it)][block_name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    ns_print = ns_val if ns_val is not None else ns_jx
                    return ns_print, it, idx, max_abs, max_rel
            return None

        def _first_gc_mismatch() -> tuple[int | None, int, int, float, float] | None:
            vm_raw = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
            jx_raw = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "raw"}
            common = sorted(vm_raw.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
            for ns_val, it in common:
                ns_jx, jx_data = _resolve_other(jx_raw, ns=ns_val, it=it)
                if jx_data is None:
                    continue
                vm = np.asarray(vm_raw[(ns_val, it)]).ravel()
                jx = np.asarray(jx_data).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    ns_print = ns_val if ns_val is not None else ns_jx
                    return ns_print, it, idx, max_abs, max_rel
            return None

        def _first_kernel_mismatch(name: str) -> tuple[int | None, int, int, float, float] | None:
            common = sorted(vmec_kernels.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1]))
            for ns_val, it in common:
                ns_jx, jx_k = _resolve_other(jax_kernels, ns=ns_val, it=it)
                if jx_k is None:
                    continue
                vm = np.asarray(vmec_kernels[(ns_val, it)][name]).ravel()
                jx = np.asarray(jx_k.get(name, np.zeros_like(vmec_kernels[(ns_val, it)][name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    ns_print = ns_val if ns_val is not None else ns_jx
                    return ns_print, it, idx, max_abs, max_rel
            return None

        flsc_m = _first_block_mismatch("flsc") if (vmec_tomnsps and jax_tomnsps) else None
        gcl_m = _first_gc_mismatch() if (vmec_gc and jax_gc) else None
        blmn_m = _first_kernel_mismatch("blmn") if (vmec_kernels and jax_kernels) else None
        clmn_m = _first_kernel_mismatch("clmn") if (vmec_kernels and jax_kernels) else None

        candidates = [m for m in (flsc_m, gcl_m, blmn_m, clmn_m) if m is not None]
        if candidates:
            ns0, it0, _, _, _ = min(candidates, key=lambda x: (x[1], -1 if x[0] is None else x[0]))
        else:
            its = []
            if vmec_tomnsps and jax_tomnsps:
                its += list(vmec_tomnsps.keys())
            if vmec_gc and jax_gc:
                its += list({(ns, it) for (stage, ns, it) in vmec_gc.keys() if stage == "raw"})
            if vmec_kernels and jax_kernels:
                its += list(vmec_kernels.keys())
            if its:
                ns0, it0 = min(its, key=lambda x: (x[1], -1 if x[0] is None else x[0]))
            else:
                ns0, it0 = None, -1

        if it0 < 0:
            print("  No overlapping dumps found for lambda-path audit.")
        else:
            if vmec_tomnsps and jax_tomnsps:
                ns_jx, jx_tomn = _resolve_other(jax_tomnsps, ns=ns0, it=it0)
                if jx_tomn is not None:
                    vm = np.asarray(vmec_tomnsps[(ns0, it0)]["flsc"])
                    jx = np.asarray(jx_tomn.get("flsc", np.zeros_like(vm)))
                    max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                    decode = _decode_tomnsps_index(idx, vm.shape)
                    ns_print = ns0 if ns0 is not None else ns_jx
                    ns_tag = f"ns={ns_print} " if ns_print is not None else ""
                    print(f"  {ns_tag}iter {it0:03d} flsc: max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={decode}")
            if vmec_gc and jax_gc:
                vm_raw = {(ns, it): vals[2] for (stage, ns, it), vals in vmec_gc.items() if stage == "raw"}
                jx_raw = {(ns, it): vals[2] for (stage, ns, it), vals in jax_gc.items() if stage == "raw"}
                ns_jx, jx_data = _resolve_other(jx_raw, ns=ns0, it=it0)
                if (ns0, it0) in vm_raw and jx_data is not None:
                    vm = np.asarray(vm_raw[(ns0, it0)])
                    jx = np.asarray(jx_data)
                    max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                    decode = _decode_gc_index(idx, vm.shape)
                    ns_print = ns0 if ns0 is not None else ns_jx
                    ns_tag = f"ns={ns_print} " if ns_print is not None else ""
                    print(f"  {ns_tag}iter {it0:03d} gcl(raw): max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={decode}")
            if vmec_kernels and jax_kernels:
                ns_jx, jx_k = _resolve_other(jax_kernels, ns=ns0, it=it0)
                if (ns0, it0) in vmec_kernels and jx_k is not None:
                    for name in ("blmn", "clmn"):
                        vm = np.asarray(vmec_kernels[(ns0, it0)][name])
                        jx = np.asarray(jx_k.get(name, np.zeros_like(vm)))
                        max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                        # Decode kernel index as (js, lt, lz, mpar) used in prior printout
                        ns_dim, ntheta3, nzeta, mpar = vm.shape
                        js = idx // (ntheta3 * nzeta * mpar)
                        rem = idx % (ntheta3 * nzeta * mpar)
                        lt = rem // (nzeta * mpar)
                        rem2 = rem % (nzeta * mpar)
                        lz = rem2 // mpar
                        mp = rem2 % mpar
                        ns_print = ns0 if ns0 is not None else ns_jx
                        ns_tag = f"ns={ns_print} " if ns_print is not None else ""
                        print(
                            f"  {ns_tag}iter {it0:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx=js={js} lt={lt} lz={lz} mpar={mp}"
                        )

    if vmec_gc or jax_gc:
        print()
        print("gc parity (VMEC2000 residue vs vmec_jax gc dumps):")
        stages = sorted({stage for (stage, _ns, _it) in vmec_gc.keys()})
        if not stages:
            print("  No overlapping gc dump iterations found.")
        for stage in stages:
            vm_stage = {(ns, it): vals for (st, ns, it), vals in vmec_gc.items() if st == stage}
            jx_stage = {(ns, it): vals for (st, ns, it), vals in jax_gc.items() if st == stage}
            for ns_val, it in sorted(vm_stage.keys(), key=lambda k: (k[0] is None, k[0] or -1, k[1])):
                ns_jx, jx_vals = _resolve_other(jx_stage, ns=ns_val, it=it)
                if jx_vals is None:
                    continue
                vm_gcr, vm_gcz, vm_gcl = vm_stage[(ns_val, it)]
                jx_gcr, jx_gcz, jx_gcl = jx_vals
                max_abs_r, max_rel_r, idx_r = _max_abs_rel_err(vm_gcr.ravel(), jx_gcr.ravel())
                max_abs_z, max_rel_z, idx_z = _max_abs_rel_err(vm_gcz.ravel(), jx_gcz.ravel())
                max_abs_l, max_rel_l, idx_l = _max_abs_rel_err(vm_gcl.ravel(), jx_gcl.ravel())
                ns_print = ns_val if ns_val is not None else ns_jx
                ns_tag = f"ns={ns_print} " if ns_print is not None else ""
                print(
                    f"  {stage} {ns_tag}iter {it:03d}: gcr max_abs={max_abs_r:.3e} max_rel={max_rel_r:.3e};"
                    f" gcz max_abs={max_abs_z:.3e} max_rel={max_rel_z:.3e};"
                    f" gcl max_abs={max_abs_l:.3e} max_rel={max_rel_l:.3e}"
                )
                if (max_abs_r > 0.0) and np.isfinite(max_abs_r):
                    dec = _decode_gc_index(int(idx_r), vm_gcr.shape)
                    iv = int(idx_r)
                    print(
                        f"    gcr idx={dec} vmec={float(vm_gcr.ravel()[iv]):.16e} "
                        f"jax={float(jx_gcr.ravel()[iv]):.16e}"
                    )
                if (max_abs_z > 0.0) and np.isfinite(max_abs_z):
                    dec = _decode_gc_index(int(idx_z), vm_gcz.shape)
                    iv = int(idx_z)
                    print(
                        f"    gcz idx={dec} vmec={float(vm_gcz.ravel()[iv]):.16e} "
                        f"jax={float(jx_gcz.ravel()[iv]):.16e}"
                    )
                if (max_abs_l > 0.0) and np.isfinite(max_abs_l):
                    dec = _decode_gc_index(int(idx_l), vm_gcl.shape)
                    iv = int(idx_l)
                    print(
                        f"    gcl idx={dec} vmec={float(vm_gcl.ravel()[iv]):.16e} "
                        f"jax={float(jx_gcl.ravel()[iv]):.16e}"
                    )
                if bool(args.fail_fast) and first_mismatch is None:
                    tol_r = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcr))))
                    tol_z = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcz))))
                    tol_l = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcl))))
                    if max_abs_r > tol_r or max_abs_z > tol_z or max_abs_l > tol_l:
                        raise SystemExit(2)

    if wout is not None:
        from vmec_jax.modes import vmec_mode_table
        from vmec_jax.vmec_parity import vmec_m1_internal_to_physical_signed
        from vmec_jax.wout import state_from_wout

        wout_state = state_from_wout(wout)
        rmnc_err_internal = _rel_rms(np.asarray(run.state.Rcos), np.asarray(wout_state.Rcos))
        zmns_err_internal = _rel_rms(np.asarray(run.state.Zsin), np.asarray(wout_state.Zsin))

        modes = vmec_mode_table(int(wout.mpol), int(wout.ntor))
        m_arr = np.asarray(modes.m, dtype=int)
        n_arr = np.asarray(modes.n, dtype=int)
        sqrt2 = np.sqrt(2.0)
        mscale = np.where(m_arr == 0, 1.0, sqrt2)
        nscale = np.where(np.abs(n_arr) == 0, 1.0, sqrt2)
        mode_scale = (mscale * nscale)[None, :]
        lconm1 = bool(getattr(run.static.cfg, "lconm1", True))
        Rcos_phys, Zsin_phys, _Rsin_phys, _Zcos_phys = vmec_m1_internal_to_physical_signed(
            Rcos=np.asarray(run.state.Rcos),
            Zsin=np.asarray(run.state.Zsin),
            Rsin=np.asarray(run.state.Rsin),
            Zcos=np.asarray(run.state.Zcos),
            modes=modes,
            lthreed=bool(int(wout.ntor) > 0),
            lasym=bool(getattr(wout, "lasym", False)),
            lconm1=lconm1,
        )
        rmnc_err_phys = _rel_rms(np.asarray(Rcos_phys) * mode_scale, np.asarray(wout.rmnc))
        zmns_err_phys = _rel_rms(np.asarray(Zsin_phys) * mode_scale, np.asarray(wout.zmns))
        fsq_ref = float(wout.fsqr + wout.fsqz + wout.fsql)
        fsq_new = None
        res = getattr(run, "result", None)
        if res is not None:
            fsqr_hist = getattr(res, "fsqr2_history", None)
            fsqz_hist = getattr(res, "fsqz2_history", None)
            fsql_hist = getattr(res, "fsql2_history", None)
            if fsqr_hist is not None and fsqz_hist is not None and fsql_hist is not None:
                try:
                    fsq_new = float(np.asarray(fsqr_hist)[-1] + np.asarray(fsqz_hist)[-1] + np.asarray(fsql_hist)[-1])
                except Exception:
                    fsq_new = None
        if fsq_new is None:
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
        print(f"  rmnc relRMS (physical)={rmnc_err_phys:.3e}  zmns relRMS (physical)={zmns_err_phys:.3e}")
        print(f"  rmnc relRMS (internal)={rmnc_err_internal:.3e}  zmns relRMS (internal)={zmns_err_internal:.3e}")

        from vmec_jax.wout import wout_minimal_from_fixed_boundary

        fsqr = fsqz = fsql = None
        fsqt = None
        converged = None
        if res is not None:
            converged = getattr(res, "diagnostics", {}).get("converged", None)
            fsqr_hist = getattr(res, "fsqr2_history", None)
            fsqz_hist = getattr(res, "fsqz2_history", None)
            fsql_hist = getattr(res, "fsql2_history", None)
            if fsqr_hist is not None and fsqz_hist is not None:
                fsqr_hist = np.asarray(fsqr_hist, dtype=float)
                fsqz_hist = np.asarray(fsqz_hist, dtype=float)
                fsqt_hist = fsqr_hist + fsqz_hist
                nstore = 100
                niter = int(fsqt_hist.size)
                stride = int(niter // nstore) + 1 if niter > 0 else 1
                fsqt = np.zeros((nstore,), dtype=float)
                count = 0
                for iter2 in range(1, niter + 1):
                    if iter2 % stride != 0:
                        continue
                    fsqt[count] = float(fsqt_hist[iter2 - 1])
                    count += 1
                    if count >= nstore:
                        break
            if fsqr_hist is not None and fsqz_hist is not None and fsql_hist is not None:
                try:
                    fsqr = float(np.asarray(fsqr_hist)[-1])
                    fsqz = float(np.asarray(fsqz_hist)[-1])
                    fsql = float(np.asarray(fsql_hist)[-1])
                except Exception:
                    fsqr = fsqz = fsql = None
        if fsqr is None or fsqz is None or fsql is None:
            fsqr, fsqz, fsql = vj.residual_scalars_from_state(
                state=run.state,
                static=run.static,
                indata=run.indata,
                signgs=int(run.signgs),
                use_vmec_synthesis=True,
            )

        wout_jax = wout_minimal_from_fixed_boundary(
            path=Path(workdir) / "_wout_compare.nc",
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            fsqr=float(fsqr),
            fsqz=float(fsqz),
            fsql=float(fsql),
            fsqt=fsqt,
            converged=converged,
        )

        radial_skip = max(int(args.radial_skip), 0)
        radial_drop_edge = bool(args.radial_drop_edge)

        def _radial_trim(arr: np.ndarray) -> np.ndarray:
            out = arr
            if arr.ndim == 1:
                out = arr[radial_skip:]
            elif arr.ndim >= 2:
                out = arr[radial_skip:, ...]
            if radial_drop_edge and out.shape[0] > 0:
                out = out[:-1, ...]
            return out

        edge_note = " + edge" if radial_drop_edge else ""
        def _align_trimmed(vm_arr: np.ndarray, jx_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if vm_arr.shape != jx_arr.shape:
                min_shape = tuple(min(a, b) for a, b in zip(vm_arr.shape, jx_arr.shape))
                vm_arr = vm_arr[tuple(slice(0, n) for n in min_shape)]
                jx_arr = jx_arr[tuple(slice(0, n) for n in min_shape)]
            return vm_arr, jx_arr

        def _print_wout_metric(name: str) -> None:
            if not hasattr(wout, name) or not hasattr(wout_jax, name):
                return
            vm_arr = _radial_trim(np.asarray(getattr(wout, name)))
            jx_arr = _radial_trim(np.asarray(getattr(wout_jax, name)))
            vm_arr, jx_arr = _align_trimmed(vm_arr, jx_arr)
            if vm_arr.size == 0 or jx_arr.size == 0:
                return
            max_abs, max_rel, _ = _max_abs_rel_err(vm_arr.ravel(), jx_arr.ravel())
            print(f"    {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

        print(f"  wout parity (skip first {radial_skip} radial points{edge_note}):")
        for name in ("jdotb", "DMerc", "Dgeod", "bsubsmns", "bsubsmnc", "bmnc", "bmns"):
            _print_wout_metric(name)

        if hasattr(wout, "betapol") and hasattr(wout_jax, "betapol"):
            print(f"    betapol: vmec={float(wout.betapol):.6e} jax={float(wout_jax.betapol):.6e}")
        if hasattr(wout, "betator") and hasattr(wout_jax, "betator"):
            print(f"    betator: vmec={float(wout.betator):.6e} jax={float(wout_jax.betator):.6e}")

    if bool(args.fail_fast) and first_mismatch is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

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
_RE_XC = re.compile(r"xc_.*_ns(\d+)_iter(\d+)\.dat$")
_RE_BSUBE = re.compile(r"bsube_iter(\d+)\.dat$")
_RE_GC = re.compile(r"gc_(raw|precond)_iter(\d+)\.dat$")
_RE_TOMNSPS = re.compile(r"tomnsps_(raw|precond)?_?iter(\d+)\.dat$")
_RE_TOMNSPS_KERNELS = re.compile(r"tomnsps_kernels_iter(\d+)\.dat$")
_RE_FORCE_KERNELS = re.compile(r"force_kernels_(raw|precond)?_?iter(\d+)\.npz$")
_RE_SCALARS = re.compile(r"scalars_iter(\d+)\.dat$")
_RE_GCX2 = re.compile(r"gcx2_iter(\d+)\.dat$")
_RE_FSQ1 = re.compile(r"fsq1_iter(\d+)\.dat$")


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
        xc_vals.append(float(toks[1].replace("D", "E").replace("d", "E")))
        v_vals.append(float(toks[2].replace("D", "E").replace("d", "E")))
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


def _collect_vmec_gc_dumps(path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gc_*_iter*.dat")):
        m = _RE_GC.search(p.name)
        if not m:
            continue
        stage = m.group(1)
        it = int(m.group(2))
        out[(stage, it)] = _parse_vmec_gc_dump(p)
    return out


def _collect_jax_gc_dumps(path: Path) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gc_*_iter*.npz")):
        m = _RE_GC.search(p.name.replace(".npz", ".dat"))
        if not m:
            continue
        stage = m.group(1)
        it = int(m.group(2))
        data = np.load(p)
        gcr = np.asarray(data["gcr"])
        gcz = np.asarray(data["gcz"])
        gcl = np.asarray(data["gcl"])
        if gcr.ndim == 4:
            gcr = np.transpose(gcr, (0, 2, 1, 3))
            gcz = np.transpose(gcz, (0, 2, 1, 3))
            gcl = np.transpose(gcl, (0, 2, 1, 3))
        out[(stage, it)] = (gcr, gcz, gcl)
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


def _collect_vmec_tomnsps_dumps(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps_iter*.dat")):
        m = _RE_TOMNSPS.search(p.name)
        if not m:
            continue
        it = int(m.group(2))
        out[it] = _parse_vmec_tomnsps_dump(p)
    return out


def _collect_vmec_tomnsps_kernels_dumps(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps_kernels_iter*.dat")):
        m = _RE_TOMNSPS_KERNELS.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        out[it] = _parse_vmec_tomnsps_kernels_dump(p)
    return out


def _collect_jax_tomnsps_dumps(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("tomnsps_raw_iter*.npz")):
        m = _RE_TOMNSPS.search(p.name.replace(".npz", ".dat"))
        if not m:
            continue
        it = int(m.group(2))
        data = np.load(p)
        frcc = np.asarray(data["frcc"])
        shape = frcc.shape
        def _block(name: str) -> np.ndarray:
            arr = np.asarray(data[name])
            if arr.size == 0:
                return np.zeros(shape, dtype=frcc.dtype)
            return arr
        out[it] = {
            "frcc": frcc,
            "frss": _block("frss"),
            "fzsc": _block("fzsc"),
            "fzcs": _block("fzcs"),
            "flsc": _block("flsc"),
            "flcs": _block("flcs"),
        }
    return out


def _collect_jax_force_kernels(path: Path) -> dict[int, dict[str, np.ndarray]]:
    out: dict[int, dict[str, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("force_kernels_raw_iter*.npz")):
        m = _RE_FORCE_KERNELS.search(p.name)
        if not m:
            continue
        it = int(m.group(2))
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

        out[it] = {
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


def _collect_scalars_dumps(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("scalars_iter*.dat")):
        m = _RE_SCALARS.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        out[it] = _parse_scalars_dump(p)
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


def _collect_gcx2_dumps(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("gcx2_iter*.dat")):
        m = _RE_GCX2.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        out[it] = _parse_gcx2_dump(p)
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


def _collect_fsq1_dumps(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("fsq1_iter*.dat")):
        m = _RE_FSQ1.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        out[it] = _parse_fsq1_dump(p)
    return out


def _compute_fsq_from_dumps(
    *,
    scalars: dict[int, dict[str, float]],
    gcx2: dict[int, dict[str, float]],
    r1: float,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not scalars or not gcx2:
        return out
    scalar_items = sorted(scalars.items())
    scalar_iters = [it for it, _ in scalar_items]
    sc_idx = 0
    current_sc: dict[str, float] | None = None
    for it in sorted(gcx2.keys()):
        while sc_idx < len(scalar_iters) and scalar_iters[sc_idx] <= it:
            current_sc = scalar_items[sc_idx][1]
            sc_idx += 1
        if current_sc is None:
            continue
        sc = current_sc
        gc = gcx2[it]
        fnorm = float(sc.get("fnorm", float("nan")))
        fnormL = float(sc.get("fnormL", float("nan")))
        gcr2 = float(gc.get("gcr2", float("nan")))
        gcz2 = float(gc.get("gcz2", float("nan")))
        gcl2 = float(gc.get("gcl2", float("nan")))
        out[it] = {
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


def _collect_bsube_dumps(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not path.exists():
        return out
    for p in sorted(path.glob("bsube_iter*.dat")):
        m = _RE_BSUBE.search(p.name)
        if not m:
            continue
        it = int(m.group(1))
        out[it] = _parse_bsube_dump(p)
    return out


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


def _decode_xc_index(idx: int, *, ns: int, mpol: int, ntor: int, lthreed: bool) -> str:
    """Decode xc/xcdot flat index into (component, m, n, js) info."""
    ns = int(ns)
    mpol = int(mpol)
    ntor = int(ntor)
    ntmax = 2 if bool(lthreed) else 1
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
    if ntmax == 1:
        comps = ("rcc", "zsc", "lsc")
    else:
        comps = ("rcc", "zsc", "lsc", "rss", "zcs", "lcs")
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
    p.add_argument(
        "--dump-level",
        choices=("full", "lite"),
        default="full",
        help=(
            "Control VMEC2000 dump verbosity. "
            "'full' enables all dumps; 'lite' keeps scalar/trace dumps only."
        ),
    )
    p.add_argument(
        "--vmec-timeout",
        type=float,
        default=None,
        help="Timeout (seconds) for the VMEC2000 run. Default: no timeout.",
    )
    args = p.parse_args()
    vmec_fsq_dump: dict[int, dict[str, float]] = {}
    jax_fsq_dump: dict[int, dict[str, float]] = {}
    vmec_fsq1: dict[int, dict[str, float]] = {}

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
        cmd = [str(vmec2000_exe), input_local.name]
        vmec_env = os.environ.copy()
        vmec_env["VMEC_DUMP_DIR"] = str(vmec_dump_dir)
        vmec_env["VMEC_DUMP_SCALARS"] = "1"
        vmec_env["VMEC_DUMP_GCX2"] = "1"
        vmec_env["VMEC_DUMP_FSQ1"] = "1"
        if args.dump_level == "full":
            vmec_env["VMEC_DUMP_XC"] = "1"
            vmec_env["VMEC_DUMP_BSUBE"] = "1"
            vmec_env["VMEC_DUMP_TOMNSPS"] = "1"
            vmec_env["VMEC_DUMP_TOMNSPS_KERNELS"] = "1"
            vmec_env["VMEC_DUMP_GC"] = "1"
            vmec_env["VMEC_DUMP_GC_STAGE"] = "both"
            vmec_env["VMEC_DUMP_GC_DIR"] = str(vmec_dump_dir)
        vmec_env.pop("VMEC_DUMP_XC_ITER", None)
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
        jax_env_backup = os.environ.copy()
        os.environ["VMEC_JAX_DUMP_XC"] = "1"
        os.environ["VMEC_JAX_DUMP_DIR"] = str(jax_dump_dir)
        os.environ["VMEC_JAX_DUMP_BSUBE"] = "1"
        os.environ["VMEC_JAX_DUMP_TOMNSPS"] = "1"
        os.environ["VMEC_JAX_DUMP_FORCE_KERNELS"] = "1"
        os.environ["VMEC_JAX_DUMP_SCALARS"] = "1"
        os.environ["VMEC_JAX_DUMP_GCX2"] = "1"
        os.environ["VMEC_JAX_DUMP_GC"] = "1"
        os.environ["VMEC_JAX_DUMP_GC_STAGE"] = "both"
        os.environ["VMEC_JAX_DUMP_GC_DIR"] = str(jax_dump_dir)
        os.environ.pop("VMEC_JAX_DUMP_ITER", None)
        try:
            run = vj.run_fixed_boundary(
                input_path,
                solver="vmec2000_iter",
                max_iter=int(args.max_iter),
                multigrid_use_input_niter=bool(args.use_input_niter),
                verbose=False,
                ns_override=int(args.single_ns) if args.single_ns is not None else None,
            )
        finally:
            os.environ.clear()
            os.environ.update(jax_env_backup)

        vmec_xc = _collect_vmec_xc_dumps(vmec_dump_dir)
        jax_xc = _collect_jax_xc_dumps(jax_dump_dir)
        vmec_bsube = _collect_bsube_dumps(vmec_dump_dir)
        jax_bsube = _collect_bsube_dumps(jax_dump_dir)
        vmec_gc = _collect_vmec_gc_dumps(vmec_dump_dir)
        jax_gc = _collect_jax_gc_dumps(jax_dump_dir)
        vmec_tomnsps = _collect_vmec_tomnsps_dumps(vmec_dump_dir)
        jax_tomnsps = _collect_jax_tomnsps_dumps(jax_dump_dir)
        vmec_kernels = _collect_vmec_tomnsps_kernels_dumps(vmec_dump_dir)
        jax_kernels = _collect_jax_force_kernels(jax_dump_dir)
        vmec_scalars = _collect_scalars_dumps(vmec_dump_dir)
        jax_scalars = _collect_scalars_dumps(jax_dump_dir)
        vmec_gcx2 = _collect_gcx2_dumps(vmec_dump_dir)
        jax_gcx2 = _collect_gcx2_dumps(jax_dump_dir)
        vmec_fsq1 = _collect_fsq1_dumps(vmec_dump_dir)
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
    vmec_niter = np.asarray([int(st.niter) for st in vmec_stages], dtype=int)
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
                vmec_fsqr = float(row.fsqr)
                vmec_fsqz = float(row.fsqz)
                vmec_fsql = float(row.fsql)
                vmec_fsqr1 = float(row.fsqr1)
                vmec_fsqz1 = float(row.fsqz1)
                vmec_fsql1 = float(row.fsql1)
                global_it = int(vmec_offsets[stage_i]) + int(row.it)
                if vmec_fsq_dump and global_it in vmec_fsq_dump:
                    dump_vals = vmec_fsq_dump[global_it]
                    vmec_fsqr = float(dump_vals.get("fsqr", vmec_fsqr))
                    vmec_fsqz = float(dump_vals.get("fsqz", vmec_fsqz))
                    vmec_fsql = float(dump_vals.get("fsql", vmec_fsql))
                if vmec_fsq1 and global_it in vmec_fsq1:
                    dump_vals = vmec_fsq1[global_it]
                    vmec_fsqr1 = float(dump_vals.get("fsqr1", vmec_fsqr1))
                    vmec_fsqz1 = float(dump_vals.get("fsqz1", vmec_fsqz1))
                    vmec_fsql1 = float(dump_vals.get("fsql1", vmec_fsql1))
                print(
                    f"  {stage_i+1:>3d} {row.it:>4d}  "
                    f"{vmec_fsqr:>11.3e} {fsqr[j] if j < fsqr.size else float('nan'):>11.3e}  "
                    f"{vmec_fsqz:>11.3e} {fsqz[j] if j < fsqz.size else float('nan'):>11.3e}  "
                    f"{vmec_fsql:>11.3e} {fsql[j] if j < fsql.size else float('nan'):>11.3e}  "
                    f"{vmec_fsqr1:>11.3e} {fsqr1[j] if j < fsqr1.size else float('nan'):>11.3e}  "
                    f"{vmec_fsqz1:>11.3e} {fsqz1[j] if j < fsqz1.size else float('nan'):>11.3e}  "
                    f"{vmec_fsql1:>11.3e} {fsql1[j] if j < fsql1.size else float('nan'):>11.3e}  "
                    f"{(row.delt0r if row.delt0r is not None else float('nan')):>11.3e} {delt[j] if j < delt.size else float('nan'):>11.3e}  "
                    f"{(row.r00 if row.r00 is not None else float('nan')):>11.3e} {r00[j] if j < r00.size else float('nan'):>11.3e}  "
                    f"{(row.w if row.w is not None else float('nan')):>11.3e} {w[j] if j < w.size else float('nan'):>11.3e}"
                )
                diff_rows.append((int(stage_i + 1), int(row.it)))
                diff_cols_vmec["fsqr"].append(vmec_fsqr)
                diff_cols_jax["fsqr"].append(float(fsqr[j]))
                diff_cols_vmec["fsqz"].append(vmec_fsqz)
                diff_cols_jax["fsqz"].append(float(fsqz[j]))
                diff_cols_vmec["fsql"].append(vmec_fsql)
                diff_cols_jax["fsql"].append(float(fsql[j]))
                diff_cols_vmec["fsqr1"].append(vmec_fsqr1)
                diff_cols_jax["fsqr1"].append(float(fsqr1[j] if j < fsqr1.size else float("nan")))
                diff_cols_vmec["fsqz1"].append(vmec_fsqz1)
                diff_cols_jax["fsqz1"].append(float(fsqz1[j] if j < fsqz1.size else float("nan")))
                diff_cols_vmec["fsql1"].append(vmec_fsql1)
                diff_cols_jax["fsql1"].append(float(fsql1[j] if j < fsql1.size else float("nan")))
                diff_cols_vmec["delt0r"].append(float(row.delt0r if row.delt0r is not None else float("nan")))
                diff_cols_jax["delt0r"].append(float(delt[j] if j < delt.size else float("nan")))
                diff_cols_vmec["r00"].append(float(row.r00 if row.r00 is not None else float("nan")))
                diff_cols_jax["r00"].append(float(r00[j] if j < r00.size else float("nan")))
                diff_cols_vmec["w"].append(float(row.w if row.w is not None else float("nan")))
                diff_cols_jax["w"].append(float(w[j] if j < w.size else float("nan")))

                if bool(args.fail_fast):
                    pairs = [
                        ("fsqr", vmec_fsqr, float(fsqr[j] if j < fsqr.size else float("nan"))),
                        ("fsqz", vmec_fsqz, float(fsqz[j] if j < fsqz.size else float("nan"))),
                        ("fsql", vmec_fsql, float(fsql[j] if j < fsql.size else float("nan"))),
                        ("fsqr1", vmec_fsqr1, float(fsqr1[j] if j < fsqr1.size else float("nan"))),
                        ("fsqz1", vmec_fsqz1, float(fsqz1[j] if j < fsqz1.size else float("nan"))),
                        ("fsql1", vmec_fsql1, float(fsql1[j] if j < fsql1.size else float("nan"))),
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
        vmec_global: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        jax_global: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for (ns_val, it_val), data in vmec_xc.items():
            matches = np.where(vmec_ns == int(ns_val))[0]
            if matches.size == 0:
                continue
            stage_idx = int(matches[0])
            global_it = int(vmec_offsets[stage_idx]) + int(it_val)
            if global_it not in vmec_global:
                vmec_global[global_it] = data
        for (ns_val, it_val), data in jax_xc.items():
            matches = np.where(vmec_ns == int(ns_val))[0]
            if matches.size == 0:
                continue
            stage_idx = int(matches[0])
            global_it = int(vmec_offsets[stage_idx]) + int(it_val)
            if global_it not in jax_global:
                jax_global[global_it] = data
        common = sorted(set(vmec_global.keys()) & set(jax_global.keys()))
        if not common:
            print("  No overlapping xc dump iterations found.")
        for it in common:
            vm_xc, vm_v = vmec_global[it]
            jx_xc, jx_v = jax_global[it]
            ok_xc, msg_xc, idx_xc = _compare_vectors(
                label="xc",
                vmec_vec=vm_xc,
                jax_vec=jx_xc,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
            ok_v, msg_v, idx_v = _compare_vectors(
                label="v",
                vmec_vec=vm_v,
                jax_vec=jx_v,
                rtol=float(args.rtol),
                atol=float(args.atol),
            )
            print(f"  iter {it:03d}: {msg_xc}; {msg_v}")
            if not ok_xc or not ok_v:
                try:
                    cfg = run.cfg
                    dec_xc = _decode_xc_index(
                        int(idx_xc),
                        ns=int(cfg.ns),
                        mpol=int(cfg.mpol),
                        ntor=int(cfg.ntor),
                        lthreed=bool(cfg.lthreed),
                    )
                except Exception:
                    dec_xc = "idx decode unavailable"
                try:
                    cfg = run.cfg
                    dec_v = _decode_xc_index(
                        int(idx_v),
                        ns=int(cfg.ns),
                        mpol=int(cfg.mpol),
                        ntor=int(cfg.ntor),
                        lthreed=bool(cfg.lthreed),
                    )
                except Exception:
                    dec_v = "idx decode unavailable"
                print(f"    xc decode: {dec_xc}")
                print(f"    v decode: {dec_v}")
            if bool(args.fail_fast) and (not ok_xc or not ok_v):
                raise SystemExit(2)

    if vmec_bsube or jax_bsube:
        print()
        print("bsube parity (VMEC2000 bcovar vs vmec_jax bcovar):")
        common = sorted(set(vmec_bsube.keys()) & set(jax_bsube.keys()))
        if not common:
            print("  No overlapping bsube dump iterations found.")
        for it in common:
            vm_bsubu, vm_bsubv = vmec_bsube[it]
            jx_bsubu, jx_bsubv = jax_bsube[it]
            max_abs_u, max_rel_u, _ = _max_abs_rel_err(vm_bsubu.ravel(), jx_bsubu.ravel())
            max_abs_v, max_rel_v, _ = _max_abs_rel_err(vm_bsubv.ravel(), jx_bsubv.ravel())
            print(
                f"  iter {it:03d}: bsubu max_abs={max_abs_u:.3e} max_rel={max_rel_u:.3e};"
                f" bsubv max_abs={max_abs_v:.3e} max_rel={max_rel_v:.3e}"
            )
            if bool(args.fail_fast):
                tol_u = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubu))))
                tol_v = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_bsubv))))
                if max_abs_u > tol_u or max_abs_v > tol_v:
                    raise SystemExit(2)

    if vmec_scalars or jax_scalars:
        print()
        print("bcovar scalar parity (wb/wp/volume/r2/fnorm/fnormL):")
        common = sorted(set(vmec_scalars.keys()) & set(jax_scalars.keys()))
        if not common:
            print("  No overlapping scalars dump iterations found.")
        for it in common:
            vm = vmec_scalars[it]
            jx = jax_scalars[it]
            for name in ("wb", "wp", "volume", "r2", "fnorm", "fnormL"):
                v = float(vm.get(name, float("nan")))
                j = float(jx.get(name, float("nan")))
                max_abs = abs(v - j)
                max_rel = max_abs / max(abs(v), float(args.atol)) if np.isfinite(v) else float("nan")
                print(f"  iter {it:03d} {name}: vmec={v:.6e} jax={j:.6e} abs={max_abs:.3e} rel={max_rel:.3e}")
                if bool(args.fail_fast):
                    tol = max(float(args.atol), float(args.rtol) * abs(v))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_gcx2 or jax_gcx2:
        print()
        print("gcx2 parity (post-scalxc/m1 sums):")
        common = sorted(set(vmec_gcx2.keys()) & set(jax_gcx2.keys()))
        if not common:
            print("  No overlapping gcx2 dump iterations found.")
        for it in common:
            vm = vmec_gcx2[it]
            jx = jax_gcx2[it]
            for name in ("gcr2", "gcz2", "gcl2"):
                v = float(vm.get(name, float("nan")))
                j = float(jx.get(name, float("nan")))
                max_abs = abs(v - j)
                max_rel = max_abs / max(abs(v), float(args.atol)) if np.isfinite(v) else float("nan")
                print(f"  iter {it:03d} {name}: vmec={v:.6e} jax={j:.6e} abs={max_abs:.3e} rel={max_rel:.3e}")
                if bool(args.fail_fast):
                    tol = max(float(args.atol), float(args.rtol) * abs(v))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_gcx2 and (include_edge_hist.size or zero_m1_hist.size):
        print()
        print("gating parity (include_edge / zero_m1):")
        for it in sorted(vmec_gcx2.keys()):
            idx = it - 1
            if idx < 0:
                continue
            if include_edge_hist.size:
                vm_edge = int(round(float(vmec_gcx2[it].get("include_edge", 0.0))))
                jx_edge = int(include_edge_hist[idx]) if idx < include_edge_hist.size else -1
                print(f"  iter {it:03d} include_edge: vmec={vm_edge} jax={jx_edge}")
                if bool(args.fail_fast) and (jx_edge >= 0) and (vm_edge != jx_edge):
                    raise SystemExit(2)
            if zero_m1_hist.size and vmec_fsq_dump:
                if it < 2:
                    vm_zero_m1 = 1
                else:
                    prev = vmec_fsq_dump.get(it - 1, {})
                    fsqz_prev = float(prev.get("fsqz", 0.0))
                    vm_zero_m1 = 1 if fsqz_prev < 1.0e-6 else 0
                jx_zero_m1 = int(zero_m1_hist[idx]) if idx < zero_m1_hist.size else -1
                print(f"  iter {it:03d} zero_m1: vmec={vm_zero_m1} jax={jx_zero_m1}")
                if bool(args.fail_fast) and (jx_zero_m1 >= 0) and (vm_zero_m1 != jx_zero_m1):
                    raise SystemExit(2)

    if vmec_kernels or jax_kernels:
        print()
        print("tomnsps kernels parity (VMEC2000 vs vmec_jax force kernels):")
        common = sorted(set(vmec_kernels.keys()) & set(jax_kernels.keys()))
        if not common:
            print("  No overlapping tomnsps_kernels/force_kernels dump iterations found.")
        for it in common:
            vm = vmec_kernels[it]
            jx = jax_kernels[it]
            for name in ("blmn", "clmn"):
                v = np.asarray(vm[name]).ravel()
                j = np.asarray(jx.get(name, np.zeros_like(vm[name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(v, j)
                msg = f"  iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
                if idx >= 0:
                    msg += f" idx={_format_kernel_index(int(idx), shape=vm[name].shape)}"
                print(msg)
                if bool(args.fail_fast):
                    tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(v))))
                    if max_abs > tol:
                        raise SystemExit(2)

    if vmec_tomnsps or jax_tomnsps:
        print()
        print("tomnsps parity (VMEC2000 vs vmec_jax raw blocks):")
        common = sorted(set(vmec_tomnsps.keys()) & set(jax_tomnsps.keys()))
        if not common:
            print("  No overlapping tomnsps dump iterations found.")
        for it in common:
            vm = vmec_tomnsps[it]
            jx = jax_tomnsps[it]
            for name in ("frcc", "frss", "fzsc", "fzcs", "flsc", "flcs"):
                v = np.asarray(vm[name]).ravel()
                j = np.asarray(jx.get(name, np.zeros_like(vm[name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(v, j)
                print(f"  iter {it:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
                if bool(args.fail_fast):
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
        common = sorted(set(vmec_tomnsps.keys()) & set(jax_tomnsps.keys()))
        raw_gcl_vm = {it: vals[2] for (stage, it), vals in vmec_gc.items() if stage == "raw"}
        raw_gcl_jx = {it: vals[2] for (stage, it), vals in jax_gc.items() if stage == "raw"}
        common_gcl = sorted(set(raw_gcl_vm.keys()) & set(raw_gcl_jx.keys()))
        for it in common:
            vm_fl = np.asarray(vmec_tomnsps[it]["flsc"])
            jx_fl = np.asarray(jax_tomnsps[it].get("flsc", np.zeros_like(vm_fl)))
            max_abs, _max_rel, idx = _max_abs_rel_err(vm_fl.ravel(), jx_fl.ravel())
            denom = _rms(vm_fl)
            norm_err = max_abs / denom if denom > 0 else float("nan")
            decode = _decode_tomnsps_index(idx, vm_fl.shape)
            print(f"  iter {it:03d} flsc: rms={denom:.3e} max_abs={max_abs:.3e} norm_err={norm_err:.3e} idx={decode}")
            if bool(args.fail_fast):
                tol = max(float(args.atol), float(args.rtol) * denom)
                if max_abs > tol:
                    raise SystemExit(2)
        for it in common_gcl:
            vm_gcl = np.asarray(raw_gcl_vm[it])
            jx_gcl = np.asarray(raw_gcl_jx[it])
            max_abs, _max_rel, idx = _max_abs_rel_err(vm_gcl.ravel(), jx_gcl.ravel())
            denom = _rms(vm_gcl)
            norm_err = max_abs / denom if denom > 0 else float("nan")
            decode = _decode_gc_index(idx, vm_gcl.shape)
            print(f"  iter {it:03d} gcl(raw): rms={denom:.3e} max_abs={max_abs:.3e} norm_err={norm_err:.3e} idx={decode}")
            if bool(args.fail_fast):
                tol = max(float(args.atol), float(args.rtol) * denom)
                if max_abs > tol:
                    raise SystemExit(2)

    # Lambda-path audit: flsc/gcl vs blmn/clmn at the first mismatch index.
    if (vmec_tomnsps and jax_tomnsps) or (vmec_gc and jax_gc) or (vmec_kernels and jax_kernels):
        print()
        print("lambda-path audit (first mismatch across flsc/gcl/blmn/clmn):")

        def _first_block_mismatch(block_name: str) -> tuple[int, int, float, float] | None:
            common = sorted(set(vmec_tomnsps.keys()) & set(jax_tomnsps.keys()))
            for it in common:
                vm = np.asarray(vmec_tomnsps[it][block_name]).ravel()
                jx = np.asarray(jax_tomnsps[it].get(block_name, np.zeros_like(vmec_tomnsps[it][block_name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    return it, idx, max_abs, max_rel
            return None

        def _first_gc_mismatch() -> tuple[int, int, float, float] | None:
            vm_raw = {it: vals[2] for (stage, it), vals in vmec_gc.items() if stage == "raw"}
            jx_raw = {it: vals[2] for (stage, it), vals in jax_gc.items() if stage == "raw"}
            common = sorted(set(vm_raw.keys()) & set(jx_raw.keys()))
            for it in common:
                vm = np.asarray(vm_raw[it]).ravel()
                jx = np.asarray(jx_raw[it]).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    return it, idx, max_abs, max_rel
            return None

        def _first_kernel_mismatch(name: str) -> tuple[int, int, float, float] | None:
            common = sorted(set(vmec_kernels.keys()) & set(jax_kernels.keys()))
            for it in common:
                vm = np.asarray(vmec_kernels[it][name]).ravel()
                jx = np.asarray(jax_kernels[it].get(name, np.zeros_like(vmec_kernels[it][name]))).ravel()
                max_abs, max_rel, idx = _max_abs_rel_err(vm, jx)
                tol = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm))))
                if max_abs > tol:
                    return it, idx, max_abs, max_rel
            return None

        flsc_m = _first_block_mismatch("flsc") if (vmec_tomnsps and jax_tomnsps) else None
        gcl_m = _first_gc_mismatch() if (vmec_gc and jax_gc) else None
        blmn_m = _first_kernel_mismatch("blmn") if (vmec_kernels and jax_kernels) else None
        clmn_m = _first_kernel_mismatch("clmn") if (vmec_kernels and jax_kernels) else None

        it_candidates = [m[0] for m in (flsc_m, gcl_m, blmn_m, clmn_m) if m is not None]
        if it_candidates:
            it0 = int(min(it_candidates))
        else:
            # No mismatch above tolerance: pick the earliest common iteration for reporting.
            its = []
            if vmec_tomnsps and jax_tomnsps:
                its += list(set(vmec_tomnsps.keys()) & set(jax_tomnsps.keys()))
            if vmec_gc and jax_gc:
                its += list({it for (stage, it) in vmec_gc.keys() if stage == "raw"} & {it for (stage, it) in jax_gc.keys() if stage == "raw"})
            if vmec_kernels and jax_kernels:
                its += list(set(vmec_kernels.keys()) & set(jax_kernels.keys()))
            it0 = int(min(its)) if its else -1

        if it0 < 0:
            print("  No overlapping dumps found for lambda-path audit.")
        else:
            if vmec_tomnsps and jax_tomnsps:
                vm = np.asarray(vmec_tomnsps[it0]["flsc"])
                jx = np.asarray(jax_tomnsps[it0].get("flsc", np.zeros_like(vm)))
                max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                decode = _decode_tomnsps_index(idx, vm.shape)
                print(f"  iter {it0:03d} flsc: max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={decode}")
            if vmec_gc and jax_gc:
                vm_raw = {it: vals[2] for (stage, it), vals in vmec_gc.items() if stage == "raw"}
                jx_raw = {it: vals[2] for (stage, it), vals in jax_gc.items() if stage == "raw"}
                if it0 in vm_raw and it0 in jx_raw:
                    vm = np.asarray(vm_raw[it0])
                    jx = np.asarray(jx_raw[it0])
                    max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                    decode = _decode_gc_index(idx, vm.shape)
                    print(f"  iter {it0:03d} gcl(raw): max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx={decode}")
            if vmec_kernels and jax_kernels:
                for name in ("blmn", "clmn"):
                    vm = np.asarray(vmec_kernels[it0][name])
                    jx = np.asarray(jax_kernels[it0].get(name, np.zeros_like(vm)))
                    max_abs, max_rel, idx = _max_abs_rel_err(vm.ravel(), jx.ravel())
                    # Decode kernel index as (js, lt, lz, mpar) used in prior printout
                    ns, ntheta3, nzeta, mpar = vm.shape
                    js = idx // (ntheta3 * nzeta * mpar)
                    rem = idx % (ntheta3 * nzeta * mpar)
                    lt = rem // (nzeta * mpar)
                    rem2 = rem % (nzeta * mpar)
                    lz = rem2 // mpar
                    mp = rem2 % mpar
                    print(
                        f"  iter {it0:03d} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} idx=js={js} lt={lt} lz={lz} mpar={mp}"
                    )

    if vmec_gc or jax_gc:
        print()
        print("gc parity (VMEC2000 residue vs vmec_jax gc dumps):")
        common = sorted(set(vmec_gc.keys()) & set(jax_gc.keys()))
        if not common:
            print("  No overlapping gc dump iterations found.")
        for stage, it in common:
            vm_gcr, vm_gcz, vm_gcl = vmec_gc[(stage, it)]
            jx_gcr, jx_gcz, jx_gcl = jax_gc[(stage, it)]
            max_abs_r, max_rel_r, _ = _max_abs_rel_err(vm_gcr.ravel(), jx_gcr.ravel())
            max_abs_z, max_rel_z, _ = _max_abs_rel_err(vm_gcz.ravel(), jx_gcz.ravel())
            max_abs_l, max_rel_l, _ = _max_abs_rel_err(vm_gcl.ravel(), jx_gcl.ravel())
            print(
                f"  {stage} iter {it:03d}: gcr max_abs={max_abs_r:.3e} max_rel={max_rel_r:.3e};"
                f" gcz max_abs={max_abs_z:.3e} max_rel={max_rel_z:.3e};"
                f" gcl max_abs={max_abs_l:.3e} max_rel={max_rel_l:.3e}"
            )
            if bool(args.fail_fast):
                tol_r = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcr))))
                tol_z = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcz))))
                tol_l = max(float(args.atol), float(args.rtol) * float(np.nanmax(np.abs(vm_gcl))))
                if max_abs_r > tol_r or max_abs_z > tol_z or max_abs_l > tol_l:
                    raise SystemExit(2)

    if wout is not None:
        rmnc_err = _rel_rms(np.asarray(run.state.Rcos), np.asarray(wout.rmnc))
        zmns_err = _rel_rms(np.asarray(run.state.Zsin), np.asarray(wout.zmns))
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
        print(f"  rmnc relRMS={rmnc_err:.3e}  zmns relRMS={zmns_err:.3e}")


if __name__ == "__main__":
    main()

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

from vmec_jax.vmec2000_exec import find_vmec2000_exec


_FORTRAN_FLOAT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:[DdEe]([+-]?\d+)|([+-]\d+))?\s*$"
)


def _parse_fortran_float(token: str) -> float:
    """Parse Fortran-style floats, including missing-exp marker forms.

    Some VMEC dumps contain underflow values like ``1.0564215887228806-316``
    (missing ``E`` before the exponent sign). Accept those here.
    """

    s = str(token).strip()
    m = _FORTRAN_FLOAT_RE.match(s)
    if m is None:
        return float(s.replace("D", "E").replace("d", "e"))
    mant = m.group(1)
    exp = m.group(2) if m.group(2) is not None else m.group(3)
    if exp is None:
        return float(mant)
    return float(f"{mant}e{exp}")


def _parse_keyvals(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or ("=" not in s):
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _infer_multigrid_from_input(input_path: Path) -> bool:
    """Infer whether VMEC multigrid staging is requested by the input file."""
    try:
        from vmec_jax.namelist import read_indata

        indata = read_indata(input_path)
        ns_array = indata.get("NS_ARRAY", None)
        if isinstance(ns_array, list):
            return len(ns_array) > 1
    except Exception:
        pass
    return False


def _parse_scalpot_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    mnpd2 = int(kv.get("mnpd2", "0"))
    mnpd = int(kv.get("mnpd", str(mnpd2)))
    nuv = int(kv.get("nuv", "0"))
    nuv3 = int(kv.get("nuv3", "0"))
    source_cache_iter = int(kv.get("source_cache_iter", "-1"))
    if mnpd2 <= 0:
        raise ValueError(f"missing mnpd2 in {path}")
    section = ""
    bvec = np.zeros((mnpd2,), dtype=float)
    bvecsav = np.zeros((mnpd2,), dtype=float)
    amat_raw = np.zeros((mnpd2, mnpd2), dtype=float)
    amat = np.zeros((mnpd2, mnpd2), dtype=float)
    xmpot = np.zeros((max(0, mnpd),), dtype=np.int64)
    xnpot = np.zeros((max(0, mnpd),), dtype=np.int64)
    source_sym_cached = np.zeros((max(0, nuv3),), dtype=float)
    gsource_cached = np.zeros((max(0, nuv),), dtype=float)
    bvecns_cached_sin = np.zeros((max(0, mnpd),), dtype=float)
    bvecns_cached_cos = np.zeros((max(0, mnpd),), dtype=float)
    grpmn_analytic = np.zeros((max(0, mnpd2), max(0, nuv3)), dtype=float)
    grpmn_total = np.zeros((max(0, mnpd2), max(0, nuv3)), dtype=float)
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
                bvec[i] = _parse_fortran_float(parts[1])
        elif section == "bvecsav" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < mnpd2:
                bvecsav[i] = _parse_fortran_float(parts[1])
        elif section == "amatrix_raw" and len(parts) >= 3:
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            if (0 <= i < mnpd2) and (0 <= j < mnpd2):
                amat_raw[i, j] = _parse_fortran_float(parts[2])
        elif section == "amatrix_lu" and len(parts) >= 3:
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            if (0 <= i < mnpd2) and (0 <= j < mnpd2):
                amat[i, j] = _parse_fortran_float(parts[2])
        elif section == "xmpot_xnpot" and len(parts) >= 3:
            i = int(parts[0]) - 1
            if 0 <= i < mnpd:
                xmpot[i] = int(round(_parse_fortran_float(parts[1])))
                xnpot[i] = int(round(_parse_fortran_float(parts[2])))
        elif section == "source_sym_cached" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < source_sym_cached.size:
                source_sym_cached[i] = _parse_fortran_float(parts[1])
        elif section == "gsource_cached" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < gsource_cached.size:
                gsource_cached[i] = _parse_fortran_float(parts[1])
        elif section == "bvecns_cached" and len(parts) >= 3:
            i = int(parts[0]) - 1
            if 0 <= i < bvecns_cached_sin.size:
                bvecns_cached_sin[i] = _parse_fortran_float(parts[1])
                bvecns_cached_cos[i] = _parse_fortran_float(parts[2])
        elif section == "grpmn_analytic" and len(parts) >= 3:
            j = int(parts[0]) - 1
            i = int(parts[1]) - 1
            if 0 <= j < grpmn_analytic.shape[0] and 0 <= i < grpmn_analytic.shape[1]:
                grpmn_analytic[j, i] = _parse_fortran_float(parts[2])
        elif section == "grpmn_total" and len(parts) >= 3:
            j = int(parts[0]) - 1
            i = int(parts[1]) - 1
            if 0 <= j < grpmn_total.shape[0] and 0 <= i < grpmn_total.shape[1]:
                grpmn_total[j, i] = _parse_fortran_float(parts[2])
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "ivacskip": int(kv.get("ivacskip", "-1")),
        "mnpd2": mnpd2,
        "mnpd": mnpd,
        "nuv": nuv,
        "nuv3": nuv3,
        "source_cache_iter": source_cache_iter,
        "bvec": bvec,
        "bvecsav": bvecsav,
        "amatrix_raw": amat_raw,
        "amatrix_lu": amat,
        "xmpot": xmpot,
        "xnpot": xnpot,
        "source_sym_cached": source_sym_cached,
        "gsource_cached": gsource_cached,
        "bvecns_cached_sin": bvecns_cached_sin,
        "bvecns_cached_cos": bvecns_cached_cos,
        "grpmn_analytic": grpmn_analytic,
        "grpmn_total": grpmn_total,
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


def _iter_from_jax_scalpot(path: Path) -> int | None:
    stem = path.stem
    prefix = "scalpot_jax_iter"
    if not stem.startswith(prefix):
        return None
    try:
        return int(stem[len(prefix) :])
    except Exception:
        return None


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
        val = _parse_fortran_float(parts[1])
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


def _parse_bextern_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    nuv3 = int(kv.get("nuv3", "0"))
    section = ""
    bexu = np.zeros((max(0, nuv3),), dtype=float)
    bexv = np.zeros((max(0, nuv3),), dtype=float)
    bexn = np.zeros((max(0, nuv3),), dtype=float)
    bexni = np.zeros((max(0, nuv3),), dtype=float)
    wint = np.zeros((max(0, nuv3),), dtype=float)
    r1b = np.zeros((max(0, nuv3),), dtype=float)
    z1b = np.zeros((max(0, nuv3),), dtype=float)
    rub = np.zeros((max(0, nuv3),), dtype=float)
    rvb = np.zeros((max(0, nuv3),), dtype=float)
    zub = np.zeros((max(0, nuv3),), dtype=float)
    zvb = np.zeros((max(0, nuv3),), dtype=float)
    snr = np.zeros((max(0, nuv3),), dtype=float)
    snv = np.zeros((max(0, nuv3),), dtype=float)
    snz = np.zeros((max(0, nuv3),), dtype=float)
    drv = np.zeros((max(0, nuv3),), dtype=float)
    guu_b = np.zeros((max(0, nuv3),), dtype=float)
    guv_b = np.zeros((max(0, nuv3),), dtype=float)
    gvv_b = np.zeros((max(0, nuv3),), dtype=float)
    auu = np.zeros((max(0, nuv3),), dtype=float)
    auv = np.zeros((max(0, nuv3),), dtype=float)
    avv = np.zeros((max(0, nuv3),), dtype=float)
    brad = np.zeros((max(0, nuv3),), dtype=float)
    bphi = np.zeros((max(0, nuv3),), dtype=float)
    bz = np.zeros((max(0, nuv3),), dtype=float)
    brad_coil = np.zeros((max(0, nuv3),), dtype=float)
    bphi_coil = np.zeros((max(0, nuv3),), dtype=float)
    bz_coil = np.zeros((max(0, nuv3),), dtype=float)
    brad_axis = np.zeros((max(0, nuv3),), dtype=float)
    bphi_axis = np.zeros((max(0, nuv3),), dtype=float)
    bz_axis = np.zeros((max(0, nuv3),), dtype=float)
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
        val = _parse_fortran_float(parts[1])
        if section == "bexu" and 0 <= idx < bexu.size:
            bexu[idx] = val
        elif section == "bexv" and 0 <= idx < bexv.size:
            bexv[idx] = val
        elif section == "bexn" and 0 <= idx < bexn.size:
            bexn[idx] = val
        elif section == "bexni" and 0 <= idx < bexni.size:
            bexni[idx] = val
        elif section == "wint" and 0 <= idx < wint.size:
            wint[idx] = val
        elif section == "r1b" and 0 <= idx < r1b.size:
            r1b[idx] = val
        elif section == "z1b" and 0 <= idx < z1b.size:
            z1b[idx] = val
        elif section == "rub" and 0 <= idx < rub.size:
            rub[idx] = val
        elif section == "rvb" and 0 <= idx < rvb.size:
            rvb[idx] = val
        elif section == "zub" and 0 <= idx < zub.size:
            zub[idx] = val
        elif section == "zvb" and 0 <= idx < zvb.size:
            zvb[idx] = val
        elif section == "snr" and 0 <= idx < snr.size:
            snr[idx] = val
        elif section == "snv" and 0 <= idx < snv.size:
            snv[idx] = val
        elif section == "snz" and 0 <= idx < snz.size:
            snz[idx] = val
        elif section == "drv" and 0 <= idx < drv.size:
            drv[idx] = val
        elif section == "guu_b" and 0 <= idx < guu_b.size:
            guu_b[idx] = val
        elif section == "guv_b" and 0 <= idx < guv_b.size:
            guv_b[idx] = val
        elif section == "gvv_b" and 0 <= idx < gvv_b.size:
            gvv_b[idx] = val
        elif section == "auu" and 0 <= idx < auu.size:
            auu[idx] = val
        elif section == "auv" and 0 <= idx < auv.size:
            auv[idx] = val
        elif section == "avv" and 0 <= idx < avv.size:
            avv[idx] = val
        elif section == "brad" and 0 <= idx < brad.size:
            brad[idx] = val
        elif section == "bphi" and 0 <= idx < bphi.size:
            bphi[idx] = val
        elif section == "bz" and 0 <= idx < bz.size:
            bz[idx] = val
        elif section == "brad_coil" and 0 <= idx < brad_coil.size:
            brad_coil[idx] = val
        elif section == "bphi_coil" and 0 <= idx < bphi_coil.size:
            bphi_coil[idx] = val
        elif section == "bz_coil" and 0 <= idx < bz_coil.size:
            bz_coil[idx] = val
        elif section == "brad_axis" and 0 <= idx < brad_axis.size:
            brad_axis[idx] = val
        elif section == "bphi_axis" and 0 <= idx < bphi_axis.size:
            bphi_axis[idx] = val
        elif section == "bz_axis" and 0 <= idx < bz_axis.size:
            bz_axis[idx] = val
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "nuv3": nuv3,
        "bexu": bexu,
        "bexv": bexv,
        "bexn": bexn,
        "bexni": bexni,
        "wint": wint,
        "r1b": r1b,
        "z1b": z1b,
        "rub": rub,
        "rvb": rvb,
        "zub": zub,
        "zvb": zvb,
        "snr": snr,
        "snv": snv,
        "snz": snz,
        "drv": drv,
        "guu_b": guu_b,
        "guv_b": guv_b,
        "gvv_b": gvv_b,
        "auu": auu,
        "auv": auv,
        "avv": avv,
        "brad": brad,
        "bphi": bphi,
        "bz": bz,
        "brad_coil": brad_coil,
        "bphi_coil": bphi_coil,
        "bz_coil": bz_coil,
        "brad_axis": brad_axis,
        "bphi_axis": bphi_axis,
        "bz_axis": bz_axis,
    }


def _parse_fouri_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    mnpd = int(kv.get("mnpd", "0"))
    mnpd2 = int(kv.get("mnpd2", str(mnpd)))
    nuv3 = int(kv.get("nuv3", "0"))
    source_sym = np.zeros((max(0, nuv3),), dtype=float)
    gsource = np.zeros((max(0, nuv3),), dtype=float)
    gsource_full = None
    grpmn = np.zeros((max(0, mnpd2), max(0, nuv3)), dtype=float)
    bvec_ns_sin = np.zeros((max(0, mnpd),), dtype=float)
    bvec_ns_cos = np.zeros((max(0, mnpd),), dtype=float)
    section = ""
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
        if section == "gsource" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < gsource.size:
                gsource[i] = _parse_fortran_float(parts[1])
        elif section == "gsource_full" and len(parts) >= 2:
            if gsource_full is None:
                gsource_full = np.zeros((0,), dtype=float)
            i = int(parts[0]) - 1
            if i >= gsource_full.size:
                pad = np.zeros((i + 1 - gsource_full.size,), dtype=float)
                gsource_full = np.concatenate([gsource_full, pad], axis=0)
            gsource_full[i] = _parse_fortran_float(parts[1])
        elif section == "source_sym" and len(parts) >= 2:
            i = int(parts[0]) - 1
            if 0 <= i < source_sym.size:
                source_sym[i] = _parse_fortran_float(parts[1])
        elif section == "grpmn" and len(parts) >= 3:
            j = int(parts[0]) - 1
            i = int(parts[1]) - 1
            if (0 <= j < grpmn.shape[0]) and (0 <= i < grpmn.shape[1]):
                grpmn[j, i] = _parse_fortran_float(parts[2])
        elif section == "bvecns" and len(parts) >= 3:
            i = int(parts[0]) - 1
            if 0 <= i < bvec_ns_sin.size:
                bvec_ns_sin[i] = _parse_fortran_float(parts[1])
                bvec_ns_cos[i] = _parse_fortran_float(parts[2])
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "mnpd": mnpd,
        "mnpd2": mnpd2,
        "nuv3": nuv3,
        "gsource": gsource,
        "gsource_full": np.asarray(gsource_full, dtype=float) if gsource_full is not None else np.zeros((0,), dtype=float),
        "source_sym": source_sym,
        "grpmn": grpmn,
        "bvecns_sin": bvec_ns_sin,
        "bvecns_cos": bvec_ns_cos,
    }


def _parse_gc_dump(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ns = mpol1 = ntor = ntmax = None
    rows: list[tuple[int, int, int, int, float, float, float]] = []
    for ln in lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or s.startswith("columns:"):
            continue
        if s.startswith("ns="):
            ns = int(s.split("=", 1)[1])
            continue
        if s.startswith("mpol1="):
            mpol1 = int(s.split("=", 1)[1])
            continue
        if s.startswith("ntor="):
            ntor = int(s.split("=", 1)[1])
            continue
        if s.startswith("ntmax="):
            ntmax = int(s.split("=", 1)[1])
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        rows.append(
            (
                int(parts[0]) - 1,
                int(parts[1]),
                int(parts[2]),
                int(parts[3]) - 1,
                _parse_fortran_float(parts[4]),
                _parse_fortran_float(parts[5]),
                _parse_fortran_float(parts[6]),
            )
        )
    if ns is None or mpol1 is None or ntor is None or ntmax is None:
        raise ValueError(f"malformed gc dump: {path}")
    gcr = np.zeros((ns, ntor + 1, mpol1 + 1, ntmax), dtype=float)
    gcz = np.zeros_like(gcr)
    gcl = np.zeros_like(gcr)
    for js, m, n, t, vcr, vcz, vcl in rows:
        if js < 0 or m < 0 or n < 0 or t < 0:
            continue
        if js >= ns or n > ntor or m > mpol1 or t >= ntmax:
            continue
        gcr[js, n, m, t] = vcr
        gcz[js, n, m, t] = vcz
        gcl[js, n, m, t] = vcl
    return {
        "ns": ns,
        "mpol1": mpol1,
        "ntor": ntor,
        "ntmax": ntmax,
        "gcr": gcr,
        "gcz": gcz,
        "gcl": gcl,
    }


def _parse_freeb_coupling_dump(path: Path) -> dict[str, Any]:
    """Parse VMEC free-boundary coupling dump from funct3d."""

    lines = path.read_text(encoding="utf-8").splitlines()
    kv = _parse_keyvals(lines)
    pgcon: list[float] = []
    rbsq: list[float] = []
    dbsq: list[float] = []
    bsqvac: list[float] = []
    p1e: list[float] = []
    p1o: list[float] = []
    pzu0: list[float] = []
    pru0: list[float] = []
    for ln in lines:
        s = ln.strip()
        if (not s) or s.startswith("#") or s.startswith("cols:"):
            continue
        parts = s.split()
        if len(parts) < 9:
            continue
        try:
            _ = int(parts[0])
            pgcon.append(_parse_fortran_float(parts[1]))
            rbsq.append(_parse_fortran_float(parts[2]))
            dbsq.append(_parse_fortran_float(parts[3]))
            bsqvac.append(_parse_fortran_float(parts[4]))
            p1e.append(_parse_fortran_float(parts[5]))
            p1o.append(_parse_fortran_float(parts[6]))
            pzu0.append(_parse_fortran_float(parts[7]))
            pru0.append(_parse_fortran_float(parts[8]))
        except Exception:
            continue
    return {
        "iter2": int(kv.get("iter2", "-1")),
        "presf_ns": float(_parse_fortran_float(kv["presf_ns"])) if "presf_ns" in kv else None,
        "pgcon": np.asarray(pgcon, dtype=float),
        "rbsq": np.asarray(rbsq, dtype=float),
        "dbsq": np.asarray(dbsq, dtype=float),
        "bsqvac": np.asarray(bsqvac, dtype=float),
        "p1e": np.asarray(p1e, dtype=float),
        "p1o": np.asarray(p1o, dtype=float),
        "pzu0": np.asarray(pzu0, dtype=float),
        "pru0": np.asarray(pru0, dtype=float),
    }


def _select_vmec_amatrix_reference(
    *,
    vmec_scal: dict[str, Any],
    vmec_dump_dir: Path,
    iter_target: int,
) -> tuple[np.ndarray, str, int | None]:
    """Choose VMEC matrix reference for comparison.

    For ``ivacskip>0`` scalpot dumps, ``amatrix_raw`` may contain the cached LU
    factors (no freshly assembled raw matrix). In that case, load the latest
    ``ivacskip=0`` dump (prefer ``source_cache_iter`` when available) so JAX raw
    mode-matrix can be compared against VMEC's raw assembly.
    """

    vmec_a_raw = np.asarray(vmec_scal.get("amatrix_raw", np.zeros_like(vmec_scal["amatrix_lu"])), dtype=float)
    vmec_a_lu = np.asarray(vmec_scal["amatrix_lu"], dtype=float)
    ivacskip = int(vmec_scal.get("ivacskip", -1))
    source_cache_iter = int(vmec_scal.get("source_cache_iter", -1))

    if ivacskip == 0:
        return vmec_a_raw, "raw", None

    cand_iters: list[int] = []
    if source_cache_iter >= 0:
        cand_iters.append(source_cache_iter)

    for p in vmec_dump_dir.glob("scalpot_iter*_ivacskip0.dat"):
        m = re.search(r"scalpot_iter(\d+)_ivacskip0\\.dat$", p.name)
        if m is None:
            continue
        it = int(m.group(1))
        if it <= int(iter_target):
            cand_iters.append(it)

    if cand_iters:
        chosen = max(cand_iters)
        p = vmec_dump_dir / f"scalpot_iter{chosen}_ivacskip0.dat"
        if p.exists():
            try:
                sc = _parse_scalpot_dump(p)
                arr = np.asarray(sc.get("amatrix_raw", np.zeros_like(vmec_a_lu)), dtype=float)
                if arr.size > 0 and np.any(np.abs(arr) > 0.0):
                    return arr, "raw_reuse_from_ivacskip0", int(chosen)
            except Exception:
                pass

    return vmec_a_lu, "lu_reuse", None


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


def _max_diff_report(
    vmec_vals: np.ndarray, jax_vals: np.ndarray, *, eps: float = 1.0e-30
) -> tuple[float, float, tuple[int, ...], float, float]:
    vmec_vals = np.asarray(vmec_vals, dtype=float)
    jax_vals = np.asarray(jax_vals, dtype=float)
    diff = np.abs(vmec_vals - jax_vals)
    if diff.size == 0:
        return float("nan"), float("nan"), (), float("nan"), float("nan")
    idx_flat = int(np.argmax(diff))
    idx = tuple(int(v) for v in np.unravel_index(idx_flat, diff.shape))
    max_abs = float(diff[idx])
    vmec_v = float(vmec_vals[idx])
    jax_v = float(jax_vals[idx])
    denom = max(eps, abs(vmec_v))
    return max_abs, float(max_abs / denom), idx, vmec_v, jax_v


def _truthy_env(name: str) -> bool:
    val = os.getenv(name, "")
    return bool(val) and val != "0"


def _find_latest_dump(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _vmec_dump_patterns(iter_target: int) -> dict[str, dict[str, str]]:
    it = int(iter_target)
    return {
        "required": {
            "scalpot": f"scalpot_iter{it}_ivacskip*.dat",
            "vacuum": f"vacuum_iter{it}_ivacskip*.dat",
        },
        "optional": {
            "bextern": f"bextern_iter{it}.dat",
            "fouri": f"fouri_iter{it}.dat",
            "freeb_coupling": f"freeb_coupling_iter{it}.dat",
            "gc_raw": f"gc_raw*_iter{it}.dat",
            "gc_precond": f"gc_precond*_iter{it}.dat",
        },
    }


def _vmec_dump_inventory(vmec_dump_dir: Path, iter_target: int) -> dict[str, Any]:
    patterns = _vmec_dump_patterns(iter_target)
    out: dict[str, Any] = {"required": {}, "optional": {}}
    for kind, specs in patterns.items():
        for name, pattern in specs.items():
            matches = sorted(vmec_dump_dir.glob(pattern))
            out[kind][name] = {
                "pattern": pattern,
                "count": len(matches),
                "files": [str(p) for p in matches],
            }
    return out


def _missing_required_vmec_dumps(vmec_dump_dir: Path, iter_target: int) -> list[str]:
    inventory = _vmec_dump_inventory(vmec_dump_dir, iter_target)
    required = inventory.get("required", {})
    missing: list[str] = []
    for name in ("scalpot", "vacuum"):
        if int(required.get(name, {}).get("count", 0)) <= 0:
            missing.append(name)
    return missing


def _missing_vmec_dump_report(
    *,
    vmec_dump_dir: Path,
    iter_target: int,
    vmec_returncodes: list[int],
    vmec_exec: Path,
    input_path: Path,
    workdir: Path,
    missing_required: list[str],
) -> dict[str, Any]:
    return {
        "status": "error",
        "ok": False,
        "error": {
            "code": "missing_vmec_dumps",
            "message": (
                "VMEC2000 completed without required instrumented dump files. "
                "Use a VMEC2000 executable built with the VMEC_DUMP_* hooks for this comparator."
            ),
            "missing_required": list(missing_required),
            "instrumentation_required": True,
            "vmec_completed_successfully": bool(vmec_returncodes) and all(rc == 0 for rc in vmec_returncodes),
        },
        "iter": int(iter_target),
        "workdir": str(workdir),
        "vmec_exec": str(vmec_exec),
        "input": str(input_path),
        "vmec_dump_dir": str(vmec_dump_dir),
        "vmec_returncodes": list(vmec_returncodes),
        "vmec_dump_requirements": {
            "required": ["scalpot", "vacuum"],
            "optional": ["bextern", "fouri", "freeb_coupling", "gc_raw", "gc_precond"],
            "note": "Only required dumps are fatal; optional dumps are included when present.",
        },
        "vmec_dump_inventory": _vmec_dump_inventory(vmec_dump_dir, iter_target),
        "requested_vmec_dump_env": {
            "VMEC_DUMP_SCALPOT": "1",
            "VMEC_DUMP_BEXTERN": "1",
            "VMEC_DUMP_FOURI": "1",
            "VMEC_DUMP_FREEB_COUPLING": "1",
            "VMEC_DUMP_DIR": str(vmec_dump_dir),
            "VMEC_DUMP_ITER": str(int(iter_target)),
        },
    }


def _vmec_run_failure_report(
    *,
    vmec_returncodes: list[int],
    vmec_exec: Path,
    input_path: Path,
    workdir: Path,
    vmec_dump_dir: Path,
    iter_target: int,
    missing_required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "ok": False,
        "error": {
            "code": "vmec2000_failed",
            "message": (
                "VMEC2000 exited with a nonzero status; not treating missing dumps "
                "as an instrumentation issue."
            ),
            "returncodes": list(vmec_returncodes),
            "missing_required": list(missing_required or []),
        },
        "iter": int(iter_target),
        "workdir": str(workdir),
        "vmec_exec": str(vmec_exec),
        "input": str(input_path),
        "vmec_dump_dir": str(vmec_dump_dir),
        "vmec_returncodes": list(vmec_returncodes),
        "vmec_dump_inventory": _vmec_dump_inventory(vmec_dump_dir, iter_target),
    }


def _emit_failure(report: dict[str, Any], json_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2)
        json_path.write_text(payload, encoding="utf-8")
        print(payload)
        return
    err = report.get("error", {})
    msg = str(err.get("message") or err.get("code") or "comparator failed")
    raise SystemExit(msg)


def _gc_metric_block(vmec_arr: np.ndarray, jax_arr: np.ndarray) -> dict[str, Any]:
    if (
        np.ndim(vmec_arr) == 4
        and np.ndim(jax_arr) == 4
        and jax_arr.shape[1] == vmec_arr.shape[2]
        and jax_arr.shape[2] == vmec_arr.shape[1]
    ):
        jax_arr = np.transpose(jax_arr, (0, 2, 1, 3))
    ns = min(vmec_arr.shape[0], jax_arr.shape[0])
    nn = min(vmec_arr.shape[1], jax_arr.shape[1])
    mm = min(vmec_arr.shape[2], jax_arr.shape[2])
    tt = min(vmec_arr.shape[3], jax_arr.shape[3])
    vm = np.asarray(vmec_arr[:ns, :nn, :mm, :tt], dtype=float)
    jj = np.asarray(jax_arr[:ns, :nn, :mm, :tt], dtype=float)
    max_abs, max_rel, idx, vmec_v, jax_v = _max_diff_report(vm, jj)
    rel_by_t = [_rel(vm[..., t], jj[..., t]) for t in range(tt)]
    max_rel_by_t = [_max_diff_report(vm[..., t], jj[..., t])[1] for t in range(tt)]
    loc = None
    if len(idx) == 4:
        loc = {
            "js": int(idx[0] + 1),
            "n": int(idx[1]),
            "m": int(idx[2]),
            "t": int(idx[3] + 1),
        }
    return {
        "shape_vmec": list(vmec_arr.shape),
        "shape_jax": list(jax_arr.shape),
        "shape_cmp": [int(ns), int(nn), int(mm), int(tt)],
        "rel_raw": _rel(vm, jj),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_loc": loc,
        "vmec_at_max": vmec_v,
        "jax_at_max": jax_v,
        "rel_raw_by_t": [float(v) for v in rel_by_t],
        "max_rel_by_t": [float(v) for v in max_rel_by_t],
    }


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
        if not mg_src.exists():
            search_roots = [
                Path(__file__).resolve().parents[2] / "examples" / "data",
                Path(__file__).resolve().parents[2] / "examples_single_grid" / "data",
                Path(__file__).resolve().parents[3] / "STELLOPT" / "BENCHMARKS" / "VMEC_TEST",
                Path(__file__).resolve().parents[3] / "external",
                Path(__file__).resolve().parents[2] / "outputs",
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


def _append_indata_overrides(path: Path, lines: list[str]) -> None:
    """Append override assignments before the terminating '/' of &INDATA."""

    txt = path.read_text(encoding="utf-8")
    m_start = re.search(r"&\s*INDATA", txt, flags=re.IGNORECASE)
    if m_start is None:
        return
    m_end = re.search(r"\n\s*/\s*\n|\n\s*/\s*$", txt[m_start.end() :], flags=re.MULTILINE)
    if m_end is None:
        return
    end_abs = m_start.end() + m_end.start()
    insert = "".join(f"  {ln}\n" for ln in lines)
    txt2 = txt[:end_abs] + ("\n" if not txt[:end_abs].endswith("\n") else "") + insert + txt[end_abs:]
    path.write_text(txt2, encoding="utf-8")


def _fmt_num(v: float) -> str:
    fv = float(v)
    if abs(fv) >= 1.0 and abs(fv - round(fv)) < 1.0e-12:
        return str(int(round(fv)))
    return f"{fv:.16e}"


def _cap_multigrid_input_to_max_iter(path: Path, max_iter: int) -> None:
    """Cap VMEC NS/NITER/FTOL arrays in-place so VMEC and JAX compare same staged budget."""

    if max_iter <= 0:
        return
    from vmec_jax.namelist import read_indata

    indata = read_indata(path)
    ns_raw = indata.get("NS_ARRAY", None)
    if not isinstance(ns_raw, list) or len(ns_raw) <= 1:
        return
    ns_arr = [int(v) for v in ns_raw]
    niter_raw = indata.get("NITER_ARRAY", None)
    if isinstance(niter_raw, list) and len(niter_raw) > 0:
        nit_arr = [max(0, int(v)) for v in niter_raw[: len(ns_arr)]]
    else:
        niter_fallback = max(0, int(indata.get_int("NITER", 0)))
        nit_arr = [niter_fallback for _ in ns_arr]
    if len(nit_arr) < len(ns_arr):
        nit_arr.extend([nit_arr[-1] if nit_arr else 0] * (len(ns_arr) - len(nit_arr)))

    ftol_raw = indata.get("FTOL_ARRAY", None)
    if isinstance(ftol_raw, list) and len(ftol_raw) > 0:
        ft_arr = [float(v) for v in ftol_raw[: len(ns_arr)]]
    else:
        ftol_fallback = float(indata.get_float("FTOL", 1.0e-30))
        ft_arr = [ftol_fallback for _ in ns_arr]
    if len(ft_arr) < len(ns_arr):
        ft_arr.extend([ft_arr[-1] if ft_arr else 1.0e-30] * (len(ns_arr) - len(ft_arr)))

    rem = int(max_iter)
    nit_cap: list[int] = []
    for i, nit_v in enumerate(nit_arr):
        if rem <= 0:
            nit_cap.append(1)
            continue
        nk = min(int(nit_v), rem)
        nk_i = int(max(1, nk))
        nit_cap.append(nk_i)
        rem -= int(max(0, nk))

    lines = [
        f"NS_ARRAY = {' '.join(str(v) for v in ns_arr)},",
        f"NITER_ARRAY = {' '.join(str(v) for v in nit_cap)},",
        f"FTOL_ARRAY = {' '.join(_fmt_num(v) for v in ft_arr)},",
    ]
    _append_indata_overrides(path, lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--vmec-exec",
        type=Path,
        default=find_vmec2000_exec(root=Path(__file__).resolve().parents[3]),
    )
    p.add_argument("--iter", type=int, default=1, help="Iteration index to compare.")
    p.add_argument("--max-iter", type=int, default=2, help="vmec_jax max_iter.")
    p.add_argument(
        "--activate-fsq",
        type=float,
        default=None,
        help=(
            "Forward to vmec_jax as free_boundary_activate_fsq. Use a large "
            "value such as 1e99 to force active free-boundary coupling in "
            "short dump-to-dump diagnostics."
        ),
    )
    p.add_argument(
        "--multigrid",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="vmec_jax multigrid staging: auto (infer from NS_ARRAY), on, or off.",
    )
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
    if args.multigrid == "auto":
        use_multigrid = _infer_multigrid_from_input(run_input)
    else:
        use_multigrid = args.multigrid == "on"
    if bool(use_multigrid) and int(args.max_iter) > 0:
        _cap_multigrid_input_to_max_iter(run_input, int(args.max_iter))

    request_gc = _truthy_env("VMEC_DUMP_GC") or _truthy_env("VMEC_JAX_DUMP_GC")
    gc_stage = os.getenv("VMEC_DUMP_GC_STAGE", os.getenv("VMEC_JAX_DUMP_GC_STAGE", "precond")).lower()
    if gc_stage not in {"raw", "precond", "both"}:
        gc_stage = "precond"

    # Run VMEC2000
    env_vmec_base = os.environ.copy()
    env_vmec_base.update(
        {
            "VMEC_DUMP_SCALPOT": "1",
            "VMEC_DUMP_BEXTERN": "1",
            "VMEC_DUMP_FOURI": "1",
            "VMEC_DUMP_FREEB_COUPLING": "1",
            "VMEC_DUMP_DIR": str(vmec_dump_dir),
        }
    )
    if request_gc:
        env_vmec_base.update(
            {
                "VMEC_DUMP_GC": "1",
                "VMEC_DUMP_GC_ITER": str(int(args.iter)),
                "VMEC_DUMP_GC_STAGE": str(gc_stage),
                "VMEC_DUMP_GC_DIR": str(vmec_dump_dir),
            }
        )

    vmec_returncodes: list[int] = []
    vmec_dump_warnings: list[str] = []

    def _run_vmec_with_dump_iter(dump_iter: int) -> int:
        env_vmec = env_vmec_base.copy()
        env_vmec["VMEC_DUMP_ITER"] = str(int(dump_iter))
        completed = subprocess.run(
            [str(vmec_exec), run_input.name],
            cwd=str(workdir),
            env=env_vmec,
            check=False,
            timeout=300,
        )
        vmec_returncodes.append(int(completed.returncode))
        return int(completed.returncode)

    first_vmec_rc = _run_vmec_with_dump_iter(int(args.iter))
    missing_required = _missing_required_vmec_dumps(vmec_dump_dir, int(args.iter))
    if first_vmec_rc != 0 and missing_required:
        _emit_failure(
            _vmec_run_failure_report(
                vmec_returncodes=vmec_returncodes,
                vmec_exec=vmec_exec,
                input_path=run_input,
                workdir=workdir,
                vmec_dump_dir=vmec_dump_dir,
                iter_target=int(args.iter),
                missing_required=missing_required,
            ),
            args.json,
        )
        return 2

    vmec_scalpot_files = sorted(vmec_dump_dir.glob(f"scalpot_iter{int(args.iter)}_ivacskip*.dat"))
    if missing_required:
        _emit_failure(
            _missing_vmec_dump_report(
                vmec_dump_dir=vmec_dump_dir,
                iter_target=int(args.iter),
                vmec_returncodes=vmec_returncodes,
                vmec_exec=vmec_exec,
                input_path=run_input,
                workdir=workdir,
                missing_required=missing_required,
            ),
            args.json,
        )
        return 2
    vmec_scal = _parse_scalpot_dump(vmec_scalpot_files[0])
    # On reuse steps (`ivacskip>0`), VMEC dump for target iteration may only
    # contain LU-form matrix data. Pull the source-cache reference iteration
    # (typically ivacskip=0) so matrix-side comparisons can use VMEC raw mode
    # space when available.
    vmec_source_cache_iter = int(vmec_scal.get("source_cache_iter", -1))
    vmec_ivacskip = int(vmec_scal.get("ivacskip", -1))
    if vmec_ivacskip > 0 and vmec_source_cache_iter >= 0:
        ref = vmec_dump_dir / f"scalpot_iter{vmec_source_cache_iter}_ivacskip0.dat"
        if not ref.exists():
            source_cache_rc = _run_vmec_with_dump_iter(vmec_source_cache_iter)
            if source_cache_rc != 0 and not ref.exists():
                vmec_dump_warnings.append(
                    "VMEC source-cache rerun exited nonzero and did not write "
                    f"{ref.name}; matrix comparison will use the target dump fallback."
                )

    # Run vmec_jax
    from vmec_jax.driver import run_fixed_boundary

    old_env = os.environ.copy()
    os.environ["VMEC_JAX_DUMP_SCALPOT"] = "1"
    os.environ["VMEC_JAX_DUMP_FREEB_COUPLING"] = "1"
    # Dump all free-boundary iterations so comparator can robustly select the
    # nearest available JAX iteration when VMEC/JAX restart bookkeeping causes
    # off-by-one or sparse dump alignment at a requested iteration.
    os.environ.pop("VMEC_JAX_DUMP_ITER", None)
    os.environ["VMEC_JAX_DUMP_DIR"] = str(jax_dump_dir)
    if request_gc:
        os.environ["VMEC_JAX_DUMP_GC"] = "1"
        os.environ["VMEC_JAX_DUMP_GC_ITER"] = str(int(args.iter))
        os.environ["VMEC_JAX_DUMP_GC_STAGE"] = str(gc_stage)
        os.environ["VMEC_JAX_DUMP_GC_DIR"] = str(jax_dump_dir)
    try:
        run_fixed_boundary(
            str(run_input),
            solver="vmec2000_iter",
            max_iter=int(args.max_iter),
            multigrid=bool(use_multigrid),
            verbose=False,
            performance_mode=False,
            use_scan=False,
            free_boundary_activate_fsq=None if args.activate_fsq is None else float(args.activate_fsq),
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)

    vmec_scalpot_files = sorted(vmec_dump_dir.glob(f"scalpot_iter{int(args.iter)}_ivacskip*.dat"))
    vmec_vac_files = sorted(vmec_dump_dir.glob(f"vacuum_iter{int(args.iter)}_ivacskip*.dat"))
    vmec_bextern_files = sorted(vmec_dump_dir.glob(f"bextern_iter{int(args.iter)}.dat"))
    vmec_fouri_files = sorted(vmec_dump_dir.glob(f"fouri_iter{int(args.iter)}.dat"))
    vmec_coupling_files = sorted(vmec_dump_dir.glob(f"freeb_coupling_iter{int(args.iter)}.dat"))
    vmec_gc_files = {
        stage: _find_latest_dump(vmec_dump_dir, f"gc_{stage}*_iter{int(args.iter)}.dat")
        for stage in ("raw", "precond")
    }
    jax_npz = jax_dump_dir / f"scalpot_jax_iter{int(args.iter)}.npz"
    jax_coupling_npz = jax_dump_dir / f"freeb_coupling_iter{int(args.iter)}.npz"
    jax_gc_npz = {
        stage: _find_latest_dump(jax_dump_dir, f"gc_{stage}*_iter{int(args.iter)}.npz")
        for stage in ("raw", "precond")
    }
    missing_required = _missing_required_vmec_dumps(vmec_dump_dir, int(args.iter))
    if missing_required:
        _emit_failure(
            _missing_vmec_dump_report(
                vmec_dump_dir=vmec_dump_dir,
                iter_target=int(args.iter),
                vmec_returncodes=vmec_returncodes,
                vmec_exec=vmec_exec,
                input_path=run_input,
                workdir=workdir,
                missing_required=missing_required,
            ),
            args.json,
        )
        return 2
    jax_iter_used = int(args.iter)
    if not jax_npz.exists():
        candidates = []
        for pth in sorted(jax_dump_dir.glob("scalpot_jax_iter*.npz")):
            it = _iter_from_jax_scalpot(pth)
            if it is not None:
                candidates.append((it, pth))
        if not candidates:
            raise SystemExit(f"missing vmec_jax dump: {jax_npz}")
        # Prefer nearest not-greater iteration, else nearest greater.
        lower = [c for c in candidates if c[0] <= int(args.iter)]
        chosen = max(lower, key=lambda x: x[0]) if lower else min(candidates, key=lambda x: x[0])
        jax_iter_used, jax_npz = int(chosen[0]), chosen[1]

    vmec_scal = _parse_scalpot_dump(vmec_scalpot_files[0])
    vmec_vac = _parse_vacuum_dump(vmec_vac_files[0])
    vmec_bex = _parse_bextern_dump(vmec_bextern_files[0]) if vmec_bextern_files else None
    vmec_fouri = _parse_fouri_dump(vmec_fouri_files[0]) if vmec_fouri_files else None
    vmec_coupling = _parse_freeb_coupling_dump(vmec_coupling_files[0]) if vmec_coupling_files else None
    jax = dict(np.load(jax_npz, allow_pickle=False))
    jax_coupling = dict(np.load(jax_coupling_npz, allow_pickle=False)) if jax_coupling_npz.exists() else None

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
        "vmec_bextern_dump": str(vmec_bextern_files[0]) if vmec_bextern_files else None,
        "vmec_fouri_dump": str(vmec_fouri_files[0]) if vmec_fouri_files else None,
        "vmec_freeb_coupling_dump": str(vmec_coupling_files[0]) if vmec_coupling_files else None,
        "jax_dump": str(jax_npz),
        "jax_freeb_coupling_dump": str(jax_coupling_npz) if jax_coupling_npz.exists() else None,
        "vmec_gc_dumps": {k: (str(v) if v is not None else None) for k, v in vmec_gc_files.items()},
        "jax_gc_dumps": {k: (str(v) if v is not None else None) for k, v in jax_gc_npz.items()},
        "iter": int(args.iter),
        "jax_iter_used": int(jax_iter_used),
        "jax_multigrid": bool(use_multigrid),
        "activate_fsq": None if args.activate_fsq is None else float(args.activate_fsq),
        "vmec_returncodes": list(vmec_returncodes),
        "vmec_dump_warnings": list(vmec_dump_warnings),
        "vmec_dump_requirements": {
            "required": ["scalpot", "vacuum"],
            "optional": ["bextern", "fouri", "freeb_coupling", "gc_raw", "gc_precond"],
            "note": "Only required dumps are fatal; optional dumps are compared when present.",
        },
        "mode_map_applied": bool(mode_map is not None),
        "vmec_scalpot_meta": {
            "iter2": int(vmec_scal.get("iter2", -1)),
            "ivacskip": int(vmec_scal.get("ivacskip", -1)),
            "source_cache_iter": int(vmec_scal.get("source_cache_iter", -1)),
        },
    }

    if request_gc:
        stages = ("raw", "precond") if gc_stage == "both" else (gc_stage,)
        for stage in stages:
            vmec_gc_path = vmec_gc_files.get(stage)
            jax_gc_path = jax_gc_npz.get(stage)
            if vmec_gc_path is None or jax_gc_path is None:
                continue
            vmec_gc = _parse_gc_dump(vmec_gc_path)
            jax_gc = dict(np.load(jax_gc_path, allow_pickle=False))
            for name in ("gcr", "gcz", "gcl"):
                if name not in jax_gc:
                    continue
                out[f"gc_{stage}_{name}"] = _gc_metric_block(
                    np.asarray(vmec_gc[name], dtype=float),
                    np.asarray(jax_gc[name], dtype=float),
                )

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
        if "bvecsav" in vmec_scal:
            vmec_bvec_ns = np.asarray(vmec_scal["bvecsav"])
            nns = min(vmec_bvec_ns.size, jax_bvec.size)
            vmns = vmec_bvec_ns[:nns]
            jjns = jax_bvec[:nns]
            a_ns, rel_ns_scaled = _rel_scaled(vmns, jjns)
            out["bvec_vs_bvecsav"] = {
                "size_vmec": int(vmec_bvec_ns.size),
                "size_jax": int(jax_bvec.size),
                "size_cmp": int(nns),
                "rel_raw": _rel(vmns, jjns),
                "rel_scaled": rel_ns_scaled,
                "scale_jax_to_vmec": a_ns,
            }
            vmec_bvec_an = np.asarray(vmec_scal["bvec"]) - np.asarray(vmec_scal["bvecsav"])
            nan = min(vmec_bvec_an.size, jax_bvec.size)
            vman = vmec_bvec_an[:nan]
            jjan = jax_bvec[:nan]
            a_an, rel_an_scaled = _rel_scaled(vman, jjan)
            out["bvec_vs_analytic"] = {
                "size_vmec": int(vmec_bvec_an.size),
                "size_jax": int(jax_bvec.size),
                "size_cmp": int(nan),
                "rel_raw": _rel(vman, jjan),
                "rel_scaled": rel_an_scaled,
                "scale_jax_to_vmec": a_an,
            }

    if "bvec_mode_nonsing_sin" in jax:
        jax_ns = np.asarray(jax["bvec_mode_nonsing_sin"])
        if "bvec_mode_nonsing_cos" in jax:
            jax_ns = np.concatenate([jax_ns, np.asarray(jax["bvec_mode_nonsing_cos"])], axis=0)
        if mode_map is not None and mode_map.size > 0 and jax_ns.size >= int(mode_map.size):
            mnpd = int(mode_map.size)
            if jax_ns.size >= 2 * mnpd:
                idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                jax_ns = jax_ns[idx]
            else:
                jax_ns = jax_ns[mode_map]
        vmec_ns = np.asarray(vmec_scal.get("bvecsav", np.zeros_like(vmec_scal["bvec"])))
        n = min(vmec_ns.size, jax_ns.size)
        vm = vmec_ns[:n]
        jj = jax_ns[:n]
        a_ns, rel_ns_scaled = _rel_scaled(vm, jj)
        out["bvec_nonsing"] = {
            "size_vmec": int(vmec_ns.size),
            "size_jax": int(jax_ns.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_ns_scaled,
            "scale_jax_to_vmec": a_ns,
        }
        vmec_ns2 = None
        vmec_ns2_src = None
        if vmec_fouri is not None:
            vmec_ns2 = np.asarray(vmec_fouri["bvecns_sin"], dtype=float)
            if bool(np.any(np.asarray(vmec_fouri["bvecns_cos"], dtype=float) != 0.0)):
                vmec_ns2 = np.concatenate([vmec_ns2, np.asarray(vmec_fouri["bvecns_cos"], dtype=float)], axis=0)
            vmec_ns2_src = "fouri"
        elif vmec_scal.get("bvecns_cached_sin", np.zeros((0,), dtype=float)).size > 0:
            vmec_ns2 = np.asarray(vmec_scal["bvecns_cached_sin"], dtype=float)
            if bool(np.any(np.asarray(vmec_scal.get("bvecns_cached_cos", np.zeros((0,), dtype=float)) != 0.0))):
                vmec_ns2 = np.concatenate([vmec_ns2, np.asarray(vmec_scal["bvecns_cached_cos"], dtype=float)], axis=0)
            vmec_ns2_src = "scalpot_cached"
        if vmec_ns2 is not None:
            n2 = min(vmec_ns2.size, jax_ns.size)
            vm2 = vmec_ns2[:n2]
            jj2 = jax_ns[:n2]
            a_ns2, rel_ns2_scaled = _rel_scaled(vm2, jj2)
            out["bvec_nonsing_fouri"] = {
                "size_vmec": int(vmec_ns2.size),
                "size_jax": int(jax_ns.size),
                "size_cmp": int(n2),
                "rel_raw": _rel(vm2, jj2),
                "rel_scaled": rel_ns2_scaled,
                "scale_jax_to_vmec": a_ns2,
                "vmec_source": vmec_ns2_src,
            }

    if "bvec_mode_analytic_sin" in jax:
        jax_an = np.asarray(jax["bvec_mode_analytic_sin"])
        if "bvec_mode_analytic_cos" in jax:
            jax_an = np.concatenate([jax_an, np.asarray(jax["bvec_mode_analytic_cos"])], axis=0)
        if mode_map is not None and mode_map.size > 0 and jax_an.size >= int(mode_map.size):
            mnpd = int(mode_map.size)
            if jax_an.size >= 2 * mnpd:
                idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                jax_an = jax_an[idx]
            else:
                jax_an = jax_an[mode_map]
        vmec_an = np.asarray(vmec_scal["bvec"]) - np.asarray(vmec_scal.get("bvecsav", np.zeros_like(vmec_scal["bvec"])))
        n = min(vmec_an.size, jax_an.size)
        vm = vmec_an[:n]
        jj = jax_an[:n]
        a_an, rel_an_scaled = _rel_scaled(vm, jj)
        out["bvec_analytic"] = {
            "size_vmec": int(vmec_an.size),
            "size_jax": int(jax_an.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_an_scaled,
            "scale_jax_to_vmec": a_an,
        }

    if "potvac" in jax:
        jpot = np.asarray(jax["potvac"], dtype=float).reshape(-1)
        vpot = np.asarray(vmec_vac["potvac"], dtype=float).reshape(-1)
        n = min(vpot.size, jpot.size)
        vm = vpot[:n]
        jj = jpot[:n]
        a_pot, rel_pot_scaled = _rel_scaled(vm, jj)
        out["potvac"] = {
            "size_vmec": int(vpot.size),
            "size_jax": int(jpot.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_pot_scaled,
            "scale_jax_to_vmec": a_pot,
        }

    if "source_sym" in jax:
        jsrc = np.asarray(jax["source_sym"], dtype=float).reshape(-1)
        if vmec_fouri is not None:
            vsrc = np.asarray(vmec_fouri["source_sym"], dtype=float).reshape(-1)
            vmec_source_kind = "fouri"
        else:
            vsrc = np.asarray(vmec_scal.get("source_sym_cached", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            vmec_source_kind = "scalpot_cached"
        n = min(vsrc.size, jsrc.size)
        vm = vsrc[:n]
        jj = jsrc[:n]
        a_src, rel_src_scaled = _rel_scaled(vm, jj)
        out["source_sym"] = {
            "size_vmec": int(vsrc.size),
            "size_jax": int(jsrc.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_src_scaled,
            "scale_jax_to_vmec": a_src,
            "vmec_source": vmec_source_kind,
        }
    if "gsource_vmec" in jax:
        jsrc = np.asarray(jax["gsource_vmec"], dtype=float).reshape(-1)
        if vmec_fouri is not None:
            vsrc = np.asarray(vmec_fouri["gsource"], dtype=float).reshape(-1)
            vmec_source_kind = "fouri"
        else:
            vsrc = np.asarray(vmec_scal.get("gsource_cached", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            vmec_source_kind = "scalpot_cached"
        n = min(vsrc.size, jsrc.size)
        vm = vsrc[:n]
        jj = jsrc[:n]
        a_src, rel_src_scaled = _rel_scaled(vm, jj)
        out["gsource"] = {
            "size_vmec": int(vsrc.size),
            "size_jax": int(jsrc.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_src_scaled,
            "scale_jax_to_vmec": a_src,
            "vmec_source": vmec_source_kind,
        }
        vfull = (
            np.asarray(vmec_fouri.get("gsource_full", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            if vmec_fouri is not None
            else np.asarray(vmec_scal.get("gsource_cached", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
        )
        if vfull.size > 0 and jsrc.size > 0:
            nfull = min(vfull.size, jsrc.size)
            vmf = vfull[:nfull]
            jjf = jsrc[:nfull]
            a_full, rel_full_scaled = _rel_scaled(vmf, jjf)
            out["gsource_full"] = {
                "size_vmec": int(vfull.size),
                "size_jax": int(jsrc.size),
                "size_cmp": int(nfull),
                "rel_raw": _rel(vmf, jjf),
                "rel_scaled": rel_full_scaled,
                "scale_jax_to_vmec": a_full,
                "vmec_source": vmec_source_kind,
            }
    if "gsource_kernel" in jax:
        jsrc = np.asarray(jax["gsource_kernel"], dtype=float).reshape(-1)
        if vmec_fouri is not None:
            vsrc = np.asarray(vmec_fouri["gsource"], dtype=float).reshape(-1)
            vmec_source_kind = "fouri"
        else:
            vsrc = np.asarray(vmec_scal.get("gsource_cached", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            vmec_source_kind = "scalpot_cached"
        n = min(vsrc.size, jsrc.size)
        vm = vsrc[:n]
        jj = jsrc[:n]
        a_src, rel_src_scaled = _rel_scaled(vm, jj)
        out["gsource_kernel"] = {
            "size_vmec": int(vsrc.size),
            "size_jax": int(jsrc.size),
            "size_cmp": int(n),
            "rel_raw": _rel(vm, jj),
            "rel_scaled": rel_src_scaled,
            "scale_jax_to_vmec": a_src,
            "vmec_source": vmec_source_kind,
        }

    if vmec_fouri is not None and "grpmn_total" in jax:
        jgr = np.asarray(jax["grpmn_total"], dtype=float)
        vgr = np.asarray(vmec_fouri.get("grpmn", np.zeros((0, 0), dtype=float)), dtype=float)
        if mode_map is not None and mode_map.size > 0 and jgr.ndim == 2:
            mnpd = int(mode_map.size)
            if jgr.shape[0] >= mnpd:
                if jgr.shape[0] >= 2 * mnpd:
                    idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                else:
                    idx = mode_map
                jgr = jgr[idx, :]
        nrow = min(vgr.shape[0], jgr.shape[0])
        ncol = min(vgr.shape[1], jgr.shape[1])
        if nrow > 0 and ncol > 0:
            vm = vgr[:nrow, :ncol]
            jj = jgr[:nrow, :ncol]
            a_gr, rel_gr_scaled = _rel_scaled(vm.reshape(-1), jj.reshape(-1))
            out["grpmn_total"] = {
                "shape_vmec": list(vgr.shape),
                "shape_jax": list(jgr.shape),
                "shape_cmp": [int(nrow), int(ncol)],
                "rel_raw": _rel(vm.reshape(-1), jj.reshape(-1)),
                "rel_scaled": rel_gr_scaled,
                "scale_jax_to_vmec": a_gr,
            }

    if "grpmn_analytic" in jax:
        jgr = np.asarray(jax["grpmn_analytic"], dtype=float)
        vgr = np.asarray(vmec_scal.get("grpmn_analytic", np.zeros((0, 0), dtype=float)), dtype=float)
        if mode_map is not None and mode_map.size > 0 and jgr.ndim == 2:
            mnpd = int(mode_map.size)
            if jgr.shape[0] >= mnpd:
                if jgr.shape[0] >= 2 * mnpd:
                    idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                else:
                    idx = mode_map
                jgr = jgr[idx, :]
        nrow = min(vgr.shape[0], jgr.shape[0])
        ncol = min(vgr.shape[1], jgr.shape[1])
        if nrow > 0 and ncol > 0:
            vm = vgr[:nrow, :ncol]
            jj = jgr[:nrow, :ncol]
            a_gr, rel_gr_scaled = _rel_scaled(vm.reshape(-1), jj.reshape(-1))
            out["grpmn_analytic"] = {
                "shape_vmec": list(vgr.shape),
                "shape_jax": list(jgr.shape),
                "shape_cmp": [int(nrow), int(ncol)],
                "rel_raw": _rel(vm.reshape(-1), jj.reshape(-1)),
                "rel_scaled": rel_gr_scaled,
                "scale_jax_to_vmec": a_gr,
            }

    if "grpmn_nonsing" in jax:
        jgr = np.asarray(jax["grpmn_nonsing"], dtype=float)
        vtot = (
            np.asarray(vmec_fouri.get("grpmn", np.zeros((0, 0), dtype=float)), dtype=float)
            if vmec_fouri is not None
            else np.asarray(vmec_scal.get("grpmn_total", np.zeros((0, 0), dtype=float)), dtype=float)
        )
        vana = np.asarray(vmec_scal.get("grpmn_analytic", np.zeros((0, 0), dtype=float)), dtype=float)
        if vtot.shape == vana.shape and vtot.size > 0:
            vgr = vtot - vana
        else:
            vgr = np.zeros((0, 0), dtype=float)
        if mode_map is not None and mode_map.size > 0 and jgr.ndim == 2:
            mnpd = int(mode_map.size)
            if jgr.shape[0] >= mnpd:
                if jgr.shape[0] >= 2 * mnpd:
                    idx = np.concatenate([mode_map, mode_map + mnpd], axis=0)
                else:
                    idx = mode_map
                jgr = jgr[idx, :]
        nrow = min(vgr.shape[0], jgr.shape[0])
        ncol = min(vgr.shape[1], jgr.shape[1])
        if nrow > 0 and ncol > 0:
            vm = vgr[:nrow, :ncol]
            jj = jgr[:nrow, :ncol]
            a_gr, rel_gr_scaled = _rel_scaled(vm.reshape(-1), jj.reshape(-1))
            out["grpmn_nonsing"] = {
                "shape_vmec": list(vgr.shape),
                "shape_jax": list(jgr.shape),
                "shape_cmp": [int(nrow), int(ncol)],
                "rel_raw": _rel(vm.reshape(-1), jj.reshape(-1)),
                "rel_scaled": rel_gr_scaled,
                "scale_jax_to_vmec": a_gr,
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
        vmec_a, vmec_kind, vmec_ref_iter = _select_vmec_amatrix_reference(
            vmec_scal=vmec_scal,
            vmec_dump_dir=vmec_dump_dir,
            iter_target=int(args.iter),
        )
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
            "vmec_matrix_kind": vmec_kind,
            "vmec_matrix_ref_iter": vmec_ref_iter,
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

    if vmec_bex is not None:
        if "bexu_ext" in jax:
            ju = np.asarray(jax["bexu_ext"]).reshape(-1)
            vu = np.asarray(vmec_bex["bexu"]).reshape(-1)
            n = min(vu.size, ju.size)
            a_u, rel_u_scaled = _rel_scaled(vu[:n], ju[:n])
            out["bexu"] = {
                "size_vmec": int(vu.size),
                "size_jax": int(ju.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vu[:n], ju[:n]),
                "rel_scaled": rel_u_scaled,
                "scale_jax_to_vmec": a_u,
            }
        if "bexv_ext" in jax:
            jv = np.asarray(jax["bexv_ext"]).reshape(-1)
            vv = np.asarray(vmec_bex["bexv"]).reshape(-1)
            n = min(vv.size, jv.size)
            a_v, rel_v_scaled = _rel_scaled(vv[:n], jv[:n])
            out["bexv"] = {
                "size_vmec": int(vv.size),
                "size_jax": int(jv.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vv[:n], jv[:n]),
                "rel_scaled": rel_v_scaled,
                "scale_jax_to_vmec": a_v,
            }
        if "bexn_ext" in jax:
            jn = np.asarray(jax["bexn_ext"]).reshape(-1)
            vn = np.asarray(vmec_bex["bexn"]).reshape(-1)
            n = min(vn.size, jn.size)
            a_n, rel_n_scaled = _rel_scaled(vn[:n], jn[:n])
            out["bexn"] = {
                "size_vmec": int(vn.size),
                "size_jax": int(jn.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vn[:n], jn[:n]),
                "rel_scaled": rel_n_scaled,
                "scale_jax_to_vmec": a_n,
            }
        bexni_key = "bexni_vmec" if "bexni_vmec" in jax else ("bexni_uniform" if "bexni_uniform" in jax else None)
        if bexni_key is not None:
            jni = np.asarray(jax[bexni_key]).reshape(-1)
            vni = np.asarray(vmec_bex["bexni"]).reshape(-1)
            n = min(vni.size, jni.size)
            a_ni, rel_ni_scaled = _rel_scaled(vni[:n], jni[:n])
            out["bexni"] = {
                "size_vmec": int(vni.size),
                "size_jax": int(jni.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vni[:n], jni[:n]),
                "rel_scaled": rel_ni_scaled,
                "scale_jax_to_vmec": a_ni,
                "jax_key": bexni_key,
            }
        if "R" in jax:
            jr = np.asarray(jax["R"]).reshape(-1)
            vr = np.asarray(vmec_bex["r1b"]).reshape(-1)
            n = min(vr.size, jr.size)
            a_r, rel_r_scaled = _rel_scaled(vr[:n], jr[:n])
            out["r1b"] = {
                "size_vmec": int(vr.size),
                "size_jax": int(jr.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vr[:n], jr[:n]),
                "rel_scaled": rel_r_scaled,
                "scale_jax_to_vmec": a_r,
            }
        if "Z" in jax:
            jz = np.asarray(jax["Z"]).reshape(-1)
            vz = np.asarray(vmec_bex["z1b"]).reshape(-1)
            n = min(vz.size, jz.size)
            a_z, rel_z_scaled = _rel_scaled(vz[:n], jz[:n])
            out["z1b"] = {
                "size_vmec": int(vz.size),
                "size_jax": int(jz.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vz[:n], jz[:n]),
                "rel_scaled": rel_z_scaled,
                "scale_jax_to_vmec": a_z,
            }
        if "Ru" in jax:
            jru = np.asarray(jax["Ru"]).reshape(-1)
            vru = np.asarray(vmec_bex["rub"]).reshape(-1)
            n = min(vru.size, jru.size)
            a_ru, rel_ru_scaled = _rel_scaled(vru[:n], jru[:n])
            out["rub"] = {
                "size_vmec": int(vru.size),
                "size_jax": int(jru.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vru[:n], jru[:n]),
                "rel_scaled": rel_ru_scaled,
                "scale_jax_to_vmec": a_ru,
            }
        if "Rv" in jax:
            jrv = np.asarray(jax["Rv"]).reshape(-1)
            vrv = np.asarray(vmec_bex["rvb"]).reshape(-1)
            n = min(vrv.size, jrv.size)
            a_rv, rel_rv_scaled = _rel_scaled(vrv[:n], jrv[:n])
            out["rvb"] = {
                "size_vmec": int(vrv.size),
                "size_jax": int(jrv.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vrv[:n], jrv[:n]),
                "rel_scaled": rel_rv_scaled,
                "scale_jax_to_vmec": a_rv,
            }
        if "Zu" in jax:
            jzu = np.asarray(jax["Zu"]).reshape(-1)
            vzu = np.asarray(vmec_bex["zub"]).reshape(-1)
            n = min(vzu.size, jzu.size)
            a_zu, rel_zu_scaled = _rel_scaled(vzu[:n], jzu[:n])
            out["zub"] = {
                "size_vmec": int(vzu.size),
                "size_jax": int(jzu.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vzu[:n], jzu[:n]),
                "rel_scaled": rel_zu_scaled,
                "scale_jax_to_vmec": a_zu,
            }
        if "Zv" in jax:
            jzv_geom = np.asarray(jax["Zv"]).reshape(-1)
            vzv_geom = np.asarray(vmec_bex["zvb"]).reshape(-1)
            n = min(vzv_geom.size, jzv_geom.size)
            a_zv_geom, rel_zv_geom_scaled = _rel_scaled(vzv_geom[:n], jzv_geom[:n])
            out["zvb"] = {
                "size_vmec": int(vzv_geom.size),
                "size_jax": int(jzv_geom.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vzv_geom[:n], jzv_geom[:n]),
                "rel_scaled": rel_zv_geom_scaled,
                "scale_jax_to_vmec": a_zv_geom,
            }
        if (
            ("snr" in jax and "snv" in jax and "snz" in jax)
            or all(k in jax for k in ("R", "Ru", "Zu", "Rv", "Zv"))
        ):
            if ("snr" in jax) and ("snv" in jax) and ("snz" in jax):
                jsnr = np.asarray(jax["snr"]).reshape(-1)
                jsnv = np.asarray(jax["snv"]).reshape(-1)
                jsnz = np.asarray(jax["snz"]).reshape(-1)
            else:
                jR = np.asarray(jax["R"]).reshape(-1)
                jRu = np.asarray(jax["Ru"]).reshape(-1)
                jZu = np.asarray(jax["Zu"]).reshape(-1)
                jRv = np.asarray(jax["Rv"]).reshape(-1)
                jZv = np.asarray(jax["Zv"]).reshape(-1)
                jsnr = -jR * jZu
                jsnv = jZu * jRv - jRu * jZv
                jsnz = jR * jRu
            for name, vv, jj in (
                ("snr", np.asarray(vmec_bex["snr"]).reshape(-1), jsnr),
                ("snv", np.asarray(vmec_bex["snv"]).reshape(-1), jsnv),
                ("snz", np.asarray(vmec_bex["snz"]).reshape(-1), jsnz),
            ):
                n = min(vv.size, jj.size)
                a_t, rel_t_scaled = _rel_scaled(vv[:n], jj[:n])
                out[name] = {
                    "size_vmec": int(vv.size),
                    "size_jax": int(jj.size),
                    "size_cmp": int(n),
                    "rel_raw": _rel(vv[:n], jj[:n]),
                    "rel_scaled": rel_t_scaled,
                    "scale_jax_to_vmec": a_t,
                }
        if "br" in jax:
            jb = np.asarray(jax["br"]).reshape(-1)
            vb = np.asarray(vmec_bex["brad"]).reshape(-1)
            n = min(vb.size, jb.size)
            a_b, rel_b_scaled = _rel_scaled(vb[:n], jb[:n])
            out["brad"] = {
                "size_vmec": int(vb.size),
                "size_jax": int(jb.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vb[:n], jb[:n]),
                "rel_scaled": rel_b_scaled,
                "scale_jax_to_vmec": a_b,
            }
        if "bp" in jax:
            jp = np.asarray(jax["bp"]).reshape(-1)
            vp = np.asarray(vmec_bex["bphi"]).reshape(-1)
            n = min(vp.size, jp.size)
            a_p, rel_p_scaled = _rel_scaled(vp[:n], jp[:n])
            out["bphi"] = {
                "size_vmec": int(vp.size),
                "size_jax": int(jp.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vp[:n], jp[:n]),
                "rel_scaled": rel_p_scaled,
                "scale_jax_to_vmec": a_p,
            }
        if "bz" in jax:
            jzv = np.asarray(jax["bz"]).reshape(-1)
            vzv = np.asarray(vmec_bex["bz"]).reshape(-1)
            n = min(vzv.size, jzv.size)
            a_zv, rel_zv_scaled = _rel_scaled(vzv[:n], jzv[:n])
            out["bz"] = {
                "size_vmec": int(vzv.size),
                "size_jax": int(jzv.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vzv[:n], jzv[:n]),
                "rel_scaled": rel_zv_scaled,
                "scale_jax_to_vmec": a_zv,
            }
        if (
            "br" in jax
            and "bp" in jax
            and "bz" in jax
            and (
                ("snr" in jax and "snv" in jax and "snz" in jax)
                or all(k in jax for k in ("R", "Ru", "Zu", "Rv", "Zv"))
            )
        ):
            jbr = np.asarray(jax["br"]).reshape(-1)
            jbp = np.asarray(jax["bp"]).reshape(-1)
            jbz = np.asarray(jax["bz"]).reshape(-1)
            if ("snr" in jax) and ("snv" in jax) and ("snz" in jax):
                jsnr = np.asarray(jax["snr"]).reshape(-1)
                jsnv = np.asarray(jax["snv"]).reshape(-1)
                jsnz = np.asarray(jax["snz"]).reshape(-1)
            else:
                jR = np.asarray(jax["R"]).reshape(-1)
                jRu = np.asarray(jax["Ru"]).reshape(-1)
                jZu = np.asarray(jax["Zu"]).reshape(-1)
                jRv = np.asarray(jax["Rv"]).reshape(-1)
                jZv = np.asarray(jax["Zv"]).reshape(-1)
                jsnr = -jR * jZu
                jsnv = jZu * jRv - jRu * jZv
                jsnz = jR * jRu
            vm_snr = np.asarray(vmec_bex["snr"]).reshape(-1)
            vm_snv = np.asarray(vmec_bex["snv"]).reshape(-1)
            vm_snz = np.asarray(vmec_bex["snz"]).reshape(-1)
            vm_br = np.asarray(vmec_bex["brad"]).reshape(-1)
            vm_bp = np.asarray(vmec_bex["bphi"]).reshape(-1)
            vm_bz = np.asarray(vmec_bex["bz"]).reshape(-1)
            channels = (
                ("bexn_term_r", vm_br * vm_snr, jbr * jsnr),
                ("bexn_term_phi", vm_bp * vm_snv, jbp * jsnv),
                ("bexn_term_z", vm_bz * vm_snz, jbz * jsnz),
            )
            for name, vv, jj in channels:
                n = min(vv.size, jj.size)
                a_t, rel_t_scaled = _rel_scaled(vv[:n], jj[:n])
                out[name] = {
                    "size_vmec": int(vv.size),
                    "size_jax": int(jj.size),
                    "size_cmp": int(n),
                    "rel_raw": _rel(vv[:n], jj[:n]),
                    "rel_scaled": rel_t_scaled,
                    "scale_jax_to_vmec": a_t,
                }
        if "br_axis" in jax:
            jba = np.asarray(jax["br_axis"]).reshape(-1)
            vba = np.asarray(vmec_bex["brad_axis"]).reshape(-1)
            n = min(vba.size, jba.size)
            a_ba, rel_ba_scaled = _rel_scaled(vba[:n], jba[:n])
            out["brad_axis"] = {
                "size_vmec": int(vba.size),
                "size_jax": int(jba.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vba[:n], jba[:n]),
                "rel_scaled": rel_ba_scaled,
                "scale_jax_to_vmec": a_ba,
            }
        if "bp_axis" in jax:
            jpa = np.asarray(jax["bp_axis"]).reshape(-1)
            vpa = np.asarray(vmec_bex["bphi_axis"]).reshape(-1)
            n = min(vpa.size, jpa.size)
            a_pa, rel_pa_scaled = _rel_scaled(vpa[:n], jpa[:n])
            out["bphi_axis"] = {
                "size_vmec": int(vpa.size),
                "size_jax": int(jpa.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vpa[:n], jpa[:n]),
                "rel_scaled": rel_pa_scaled,
                "scale_jax_to_vmec": a_pa,
            }
        if "bz_axis" in jax:
            jza = np.asarray(jax["bz_axis"]).reshape(-1)
            vza = np.asarray(vmec_bex["bz_axis"]).reshape(-1)
            n = min(vza.size, jza.size)
            a_za, rel_za_scaled = _rel_scaled(vza[:n], jza[:n])
            out["bz_axis"] = {
                "size_vmec": int(vza.size),
                "size_jax": int(jza.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vza[:n], jza[:n]),
                "rel_scaled": rel_za_scaled,
                "scale_jax_to_vmec": a_za,
            }
        if "br_mgrid" in jax:
            jbc = np.asarray(jax["br_mgrid"]).reshape(-1)
            vbc = np.asarray(vmec_bex["brad_coil"]).reshape(-1)
            n = min(vbc.size, jbc.size)
            a_bc, rel_bc_scaled = _rel_scaled(vbc[:n], jbc[:n])
            out["brad_coil"] = {
                "size_vmec": int(vbc.size),
                "size_jax": int(jbc.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vbc[:n], jbc[:n]),
                "rel_scaled": rel_bc_scaled,
                "scale_jax_to_vmec": a_bc,
            }
        if "bp_mgrid" in jax:
            jpc = np.asarray(jax["bp_mgrid"]).reshape(-1)
            vpc = np.asarray(vmec_bex["bphi_coil"]).reshape(-1)
            n = min(vpc.size, jpc.size)
            a_pc, rel_pc_scaled = _rel_scaled(vpc[:n], jpc[:n])
            out["bphi_coil"] = {
                "size_vmec": int(vpc.size),
                "size_jax": int(jpc.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vpc[:n], jpc[:n]),
                "rel_scaled": rel_pc_scaled,
                "scale_jax_to_vmec": a_pc,
            }
        if "bz_mgrid" in jax:
            jzc = np.asarray(jax["bz_mgrid"]).reshape(-1)
            vzc = np.asarray(vmec_bex["bz_coil"]).reshape(-1)
            n = min(vzc.size, jzc.size)
            a_zc, rel_zc_scaled = _rel_scaled(vzc[:n], jzc[:n])
            out["bz_coil"] = {
                "size_vmec": int(vzc.size),
                "size_jax": int(jzc.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vzc[:n], jzc[:n]),
                "rel_scaled": rel_zc_scaled,
                "scale_jax_to_vmec": a_zc,
            }

    if vmec_coupling is not None and jax_coupling is not None:
        field_pairs = (
            ("pgcon", "gcon_edge"),
            ("rbsq", "rbsq_edge"),
            ("dbsq", "dbsq_edge_proxy"),
            ("bsqvac", "bsqvac_edge"),
            ("p1e", "pr1_even_edge"),
            ("p1o", "pr1_odd_edge"),
            ("pzu0", "pzu0_edge"),
            ("pru0", "pru0_edge"),
        )
        for vname, jname in field_pairs:
            if vname not in vmec_coupling or jname not in jax_coupling:
                continue
            vv = np.asarray(vmec_coupling[vname], dtype=float).reshape(-1)
            jj = np.asarray(jax_coupling[jname], dtype=float).reshape(-1)
            n = min(vv.size, jj.size)
            if n <= 0:
                continue
            a_t, rel_t_scaled = _rel_scaled(vv[:n], jj[:n])
            out[f"freeb_coupling_{vname}"] = {
                "size_vmec": int(vv.size),
                "size_jax": int(jj.size),
                "size_cmp": int(n),
                "rel_raw": _rel(vv[:n], jj[:n]),
                "rel_scaled": rel_t_scaled,
                "scale_jax_to_vmec": a_t,
            }
        try:
            v_pg = np.asarray(vmec_coupling.get("pgcon", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            v_r = np.asarray(vmec_coupling.get("rbsq", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            v_p1e = np.asarray(vmec_coupling.get("p1e", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            v_p1o = np.asarray(vmec_coupling.get("p1o", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            j_pg = np.asarray(jax_coupling.get("gcon_edge", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            j_r = np.asarray(jax_coupling.get("rbsq_edge", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            j_p1e = np.asarray(jax_coupling.get("pr1_even_edge", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            j_p1o = np.asarray(jax_coupling.get("pr1_odd_edge", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
            vn = min(v_pg.size, v_r.size, v_p1e.size, v_p1o.size)
            jn = min(j_pg.size, j_r.size, j_p1e.size, j_p1o.size)
            if vn > 0 and jn > 0:
                v_denom = v_pg[:vn] * (v_p1e[:vn] + v_p1o[:vn])
                j_denom = j_pg[:jn] * (j_p1e[:jn] + j_p1o[:jn])
                v_mask = np.abs(v_denom) > 1.0e-14
                j_mask = np.abs(j_denom) > 1.0e-14
                if np.any(v_mask) and np.any(j_mask):
                    v_ohs = np.median(v_r[:vn][v_mask] / v_denom[v_mask])
                    j_ohs = np.median(j_r[:jn][j_mask] / j_denom[j_mask])
                    out["freeb_coupling_ohs"] = {
                        "vmec": float(v_ohs),
                        "jax": float(j_ohs),
                        "ratio_jax_to_vmec": float(j_ohs / v_ohs) if abs(v_ohs) > 0.0 else None,
                        "stage_mismatch_suspected": bool(abs(v_ohs) > 0.0 and abs(j_ohs / v_ohs - 1.0) > 0.1),
                    }
        except Exception:
            pass

    print(json.dumps(out, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

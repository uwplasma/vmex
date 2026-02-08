from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import vmec_jax.api as vj


@dataclass(frozen=True)
class IterRow:
    it: int
    fsqr: float
    fsqz: float
    fsql: float
    fsqr1: float
    fsqz1: float
    fsql1: float
    delt: float


_ROW_RE = re.compile(
    r"^\s*(?P<it>\d+)\s*\|\s*"
    r"(?P<FSQR>[-+0-9.eE]+)\s+(?P<FSQZ>[-+0-9.eE]+)\s+(?P<FSQL>[-+0-9.eE]+)\s*\|\s*"
    r"(?P<fsqr1>[-+0-9.eE]+)\s+(?P<fsqz1>[-+0-9.eE]+)\s+(?P<fsql1>[-+0-9.eE]+)\s*\|\s*"
    r"(?P<DELT>[-+0-9.eE]+)\s*\|"
)


def _run_vmecpp_rows(input_file: Path, *, niter: int) -> list[IterRow]:
    try:
        import vmecpp  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("vmecpp is not importable; install/build vmecpp first") from e

    code = f"""
import vmecpp
inp=vmecpp.VmecInput.from_file({str(input_file)!r})
inp.return_outputs_even_if_not_converged=True
inp.niter_array=[{int(niter)}]
inp.ftol_array=[0.0]
try:
    inp.nstep = 1
except Exception:
    inp.scalars['NSTEP'] = 1
vmecpp.run(inp, verbose=True, max_threads=1)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    rows: list[IterRow] = []
    for line in res.stdout.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        rows.append(
            IterRow(
                it=int(m.group("it")),
                fsqr=float(m.group("FSQR")),
                fsqz=float(m.group("FSQZ")),
                fsql=float(m.group("FSQL")),
                fsqr1=float(m.group("fsqr1")),
                fsqz1=float(m.group("fsqz1")),
                fsql1=float(m.group("fsql1")),
                delt=float(m.group("DELT")),
            )
        )
    if len(rows) < niter:
        raise RuntimeError(f"parsed only {len(rows)}/{niter} iterations from vmecpp output")
    return rows[:niter]


def _run_vmecjax_rows(input_file: Path, *, niter: int) -> list[IterRow]:
    run = vj.run_fixed_boundary(input_file, solver="vmecpp_iter", max_iter=int(niter), verbose=False)
    d = run.result.diagnostics
    fsqr = np.asarray(run.result.fsqr2_history, dtype=float)
    fsqz = np.asarray(run.result.fsqz2_history, dtype=float)
    fsql = np.asarray(run.result.fsql2_history, dtype=float)
    fsqr1 = np.asarray(d["fsqr1_history"], dtype=float)
    fsqz1 = np.asarray(d["fsqz1_history"], dtype=float)
    fsql1 = np.asarray(d["fsql1_history"], dtype=float)
    delt = np.asarray(d["time_step_history"], dtype=float)
    out: list[IterRow] = []
    for i in range(min(niter, fsqr.size)):
        out.append(
            IterRow(
                it=i + 1,
                fsqr=float(fsqr[i]),
                fsqz=float(fsqz[i]),
                fsql=float(fsql[i]),
                fsqr1=float(fsqr1[i]),
                fsqz1=float(fsqz1[i]),
                fsql1=float(fsql1[i]),
                delt=float(delt[i]),
            )
        )
    return out


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / denom


def main() -> None:
    input_file = Path(__file__).resolve().parents[1] / "data" / "input.vmecpp_solovev"
    niter = 5

    vmecpp_rows = _run_vmecpp_rows(input_file, niter=niter)
    jax_rows = _run_vmecjax_rows(input_file, niter=niter)

    print(f"input={input_file}")
    print("Columns: FSQR/FSQZ/FSQL are invariant residuals; fsqr/fsqz/fsql are preconditioned residuals.")
    print("")
    print(" it | vmecpp(FSQR,FSQZ,FSQL)        | jax(FSQR,FSQZ,FSQL)          | relerr(FSQR,FSQZ,FSQL)")
    for r_pp, r_jx in zip(vmecpp_rows, jax_rows):
        e = (_rel_err(r_pp.fsqr, r_jx.fsqr), _rel_err(r_pp.fsqz, r_jx.fsqz), _rel_err(r_pp.fsql, r_jx.fsql))
        print(
            f"{r_pp.it:3d} | "
            f"{r_pp.fsqr:10.3e} {r_pp.fsqz:10.3e} {r_pp.fsql:10.3e} | "
            f"{r_jx.fsqr:10.3e} {r_jx.fsqz:10.3e} {r_jx.fsql:10.3e} | "
            f"{e[0]:9.2e} {e[1]:9.2e} {e[2]:9.2e}"
        )
    print("")
    print(" it | vmecpp(fsqr,fsqz,fsql)        | jax(fsqr,fsqz,fsql)          | relerr(fsqr,fsqz,fsql) | DELT")
    for r_pp, r_jx in zip(vmecpp_rows, jax_rows):
        e = (_rel_err(r_pp.fsqr1, r_jx.fsqr1), _rel_err(r_pp.fsqz1, r_jx.fsqz1), _rel_err(r_pp.fsql1, r_jx.fsql1))
        print(
            f"{r_pp.it:3d} | "
            f"{r_pp.fsqr1:10.3e} {r_pp.fsqz1:10.3e} {r_pp.fsql1:10.3e} | "
            f"{r_jx.fsqr1:10.3e} {r_jx.fsqz1:10.3e} {r_jx.fsql1:10.3e} | "
            f"{e[0]:9.2e} {e[1]:9.2e} {e[2]:9.2e} | "
            f"{r_pp.delt:6.2e}/{r_jx.delt:6.2e}"
        )


if __name__ == "__main__":
    main()

